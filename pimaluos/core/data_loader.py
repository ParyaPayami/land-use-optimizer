"""
PIMALUOS data loading and feature engineering.

Two loaders are provided:

* :class:`ManhattanDataLoader` / :class:`ParcelFileLoader` read a local MapPLUTO
  (or any parcel layer described by a city YAML ``column_mapping``) and build a
  standardised parcel table plus a numeric node-feature matrix.
* :class:`SyntheticCityLoader` generates a small synthetic city with the same
  schema. It exists **only** for unit tests and CI smoke runs; no result in the
  manuscript is produced from it.

All geometry is reprojected to the city's projected CRS (EPSG:2263, US feet, for
Manhattan) so areas and distances are in feet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import box

from pimaluos.config.settings import CityConfig, get_city_config

logger = logging.getLogger(__name__)

# MapPLUTO LandUse codes (two-digit strings).
LAND_USE_LABELS: Dict[str, str] = {
    "01": "one_two_family",
    "02": "multi_family_walkup",
    "03": "multi_family_elevator",
    "04": "mixed_residential_commercial",
    "05": "commercial_office",
    "06": "industrial_manufacturing",
    "07": "transportation_utility",
    "08": "public_facilities_institutions",
    "09": "open_space_recreation",
    "10": "parking",
    "11": "vacant_land",
}

# Coarse land-use class used by the capacity models.
LAND_USE_CLASS: Dict[str, str] = {
    "01": "residential", "02": "residential", "03": "residential",
    "04": "mixed", "05": "commercial", "06": "industrial",
    "07": "utility", "08": "public", "09": "open_space",
    "10": "parking", "11": "vacant",
}

RESIDENTIAL_CODES = {"01", "02", "03", "04"}
OPEN_SPACE_CODE = "09"

NUMERIC_COLUMNS = [
    "lot_area_sqft", "bldg_area_sqft", "num_floors", "year_built", "year_altered",
    "built_far", "max_resid_far", "max_comm_far", "max_facil_far", "max_manu_far", "max_affres_far",
    "assessed_total", "assessed_land", "units_res", "units_total",
    "lot_front", "lot_depth", "bldg_front", "bldg_depth",
    "res_area", "com_area", "office_area", "retail_area", "garage_area",
    "storage_area", "factory_area",
]

# Heavy-tailed raw features that are log1p-transformed before standardisation.
LOG_FEATURES = {
    "lot_area_sqft", "bldg_area_sqft", "assessed_total", "assessed_land",
    "assessed_building", "units_res", "units_total", "res_area", "com_area",
    "office_area", "retail_area", "garage_area", "storage_area", "factory_area",
    "value_per_lot_sqft", "dist_to_open_space_ft", "dist_to_center_ft",
    "perimeter_ft",
}


@dataclass
class ParcelDataset:
    """Standardised parcels plus node features.

    Attributes:
        gdf: Parcel GeoDataFrame in projected CRS with standard columns.
        features_raw: Engineered features before transformation.
        features: Transformed (log1p where appropriate) and z-scored features.
        feature_names: Column names of ``features`` (the GNN input order).
        meta: Provenance information written to the run manifest.
    """

    gdf: gpd.GeoDataFrame
    features_raw: pd.DataFrame
    features: pd.DataFrame
    feature_names: List[str]
    meta: Dict = field(default_factory=dict)


def _normalise_land_use(series: pd.Series) -> pd.Series:
    """Return MapPLUTO land-use codes as two-digit strings ('01'..'11')."""

    def conv(v):
        if pd.isna(v):
            return "11"
        try:
            return f"{int(float(v)):02d}"
        except (TypeError, ValueError):
            s = str(v).strip()
            return s.zfill(2) if s else "11"

    return series.map(conv)


def _flag(series: Optional[pd.Series], n: int) -> np.ndarray:
    """Convert a presence-type column (string or numeric) to 0/1."""
    if series is None:
        return np.zeros(n, dtype=float)
    if series.dtype == object:
        s = series.fillna("").astype(str).str.strip().str.upper()
        return (~s.isin(["", "N", "0", "NAN", "NONE"])).astype(float).values
    return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(float).values


BASE_FAR_CAPS = ["max_resid_far", "max_comm_far", "max_facil_far", "max_manu_far"]


def standardise_parcels(
    gdf: gpd.GeoDataFrame, column_mapping: Dict[str, str], target_crs: str,
    include_affordable_far: bool = False,
) -> gpd.GeoDataFrame:
    """Rename columns, coerce types and derive regulatory maxima.

    ``max_far`` is the **maximum** of the residential, commercial, community
    facility and manufacturing FAR limits (MapPLUTO ResidFAR, CommFAR, FacilFAR,
    ManuFAR). These are alternative caps for different uses, so they must not be
    summed. The higher cap available only when qualifying affordable housing is
    provided (AffResFAR, from 26v1) is excluded unless ``include_affordable_far``.
    """
    gdf = gdf.rename(columns={k: v for k, v in column_mapping.items() if k in gdf.columns})
    if gdf.crs is None:
        raise ValueError("Parcel layer has no CRS; cannot compute areas in feet.")
    gdf = gdf.to_crs(target_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.make_valid() if hasattr(gdf.geometry, "make_valid") else gdf.geometry.buffer(0)

    for col in NUMERIC_COLUMNS:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
        else:
            gdf[col] = np.nan

    # Lot area: prefer the assessor value, fall back to polygon area.
    poly_area = gdf.geometry.area
    gdf["lot_area_sqft"] = gdf["lot_area_sqft"].where(gdf["lot_area_sqft"] > 0, poly_area)
    gdf["bldg_area_sqft"] = gdf["bldg_area_sqft"].fillna(0).clip(lower=0)
    gdf["num_floors"] = gdf["num_floors"].fillna(0).clip(lower=0)
    gdf["built_far"] = gdf["built_far"].where(
        gdf["built_far"].notna(), gdf["bldg_area_sqft"] / gdf["lot_area_sqft"]
    ).fillna(0)

    for c in BASE_FAR_CAPS + ["max_affres_far"]:
        gdf[c] = gdf[c].fillna(0).clip(lower=0)
    caps = BASE_FAR_CAPS + (["max_affres_far"] if include_affordable_far else [])
    gdf["max_far"] = gdf[caps].max(axis=1)

    gdf["land_use"] = _normalise_land_use(gdf["land_use"] if "land_use" in gdf else pd.Series(np.nan, index=gdf.index))
    gdf["land_use_class"] = gdf["land_use"].map(LAND_USE_CLASS).fillna("vacant")
    gdf["zone_district"] = gdf.get("zone_district", pd.Series("UNKNOWN", index=gdf.index)).fillna("UNKNOWN").astype(str)
    gdf["address"] = gdf.get("address", pd.Series("", index=gdf.index)).fillna("").astype(str)

    centroids = gdf.geometry.centroid
    gdf["x"] = centroids.x
    gdf["y"] = centroids.y
    return gdf.reset_index(drop=True)


def compute_node_features(
    gdf: gpd.GeoDataFrame, center_xy: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Engineer node features from the standardised parcel table.

    Every feature is computed from columns present in MapPLUTO (or derived from
    geometry). Features that are constant over the study area are dropped later
    in :func:`transform_features`, so the final feature count is data-dependent
    and is recorded in the run manifest.
    """
    n = len(gdf)
    f = pd.DataFrame(index=gdf.index)

    # --- Geometry
    f["lot_area_sqft"] = gdf["lot_area_sqft"].fillna(0)
    f["perimeter_ft"] = gdf.geometry.length
    area = gdf.geometry.area.replace(0, np.nan)
    f["shape_index"] = (gdf.geometry.length ** 2 / (4 * np.pi * area)).fillna(1.0).clip(upper=50)
    for c in ["lot_front", "lot_depth", "bldg_front", "bldg_depth"]:
        f[c] = gdf[c].fillna(0)
    f["irregular_lot"] = _flag(gdf.get("irregular_lot"), n)

    # --- Built environment
    f["bldg_area_sqft"] = gdf["bldg_area_sqft"]
    f["num_floors"] = gdf["num_floors"]
    f["built_far"] = gdf["built_far"]
    floors = gdf["num_floors"].replace(0, np.nan)
    f["lot_coverage"] = (gdf["bldg_area_sqft"] / floors / gdf["lot_area_sqft"]).fillna(0).clip(0, 1)
    year_built = gdf["year_built"].where(gdf["year_built"] > 1600)
    f["year_built"] = year_built.fillna(year_built.median() if year_built.notna().any() else 1950)
    altered = gdf["year_altered"].where(gdf["year_altered"] > 1600)
    f["years_since_alteration"] = (2025 - altered.fillna(f["year_built"])).clip(lower=0)
    for c in ["units_res", "units_total", "res_area", "com_area", "office_area",
              "retail_area", "garage_area", "storage_area", "factory_area"]:
        f[c] = gdf[c].fillna(0).clip(lower=0)

    # --- Regulation
    for c in BASE_FAR_CAPS + ["max_affres_far", "max_far"]:
        f[c] = gdf[c]
    f["far_utilisation"] = (gdf["built_far"] / gdf["max_far"].replace(0, np.nan)).fillna(0).clip(0, 5)
    f["historic_district"] = _flag(gdf.get("historic_district"), n)
    f["landmark"] = _flag(gdf.get("landmark"), n)
    f["special_district"] = _flag(gdf.get("special_district"), n)
    f["split_zone"] = _flag(gdf.get("split_zone"), n)
    f["flood_zone"] = np.maximum(_flag(gdf.get("flood_2015"), n), _flag(gdf.get("flood_2007"), n))

    # --- Economics
    f["assessed_total"] = gdf["assessed_total"].fillna(0).clip(lower=0)
    f["assessed_land"] = gdf["assessed_land"].fillna(0).clip(lower=0)
    f["assessed_building"] = (f["assessed_total"] - f["assessed_land"]).clip(lower=0)
    f["value_per_lot_sqft"] = f["assessed_total"] / gdf["lot_area_sqft"].replace(0, np.nan)
    f["value_per_lot_sqft"] = f["value_per_lot_sqft"].fillna(0)

    # --- Location
    xy = gdf[["x", "y"]].values
    open_mask = (gdf["land_use"] == OPEN_SPACE_CODE).values
    if open_mask.any():
        d, _ = cKDTree(xy[open_mask]).query(xy, k=1)
        f["dist_to_open_space_ft"] = d
    else:
        f["dist_to_open_space_ft"] = 0.0
    if center_xy is not None:
        f["dist_to_center_ft"] = np.hypot(xy[:, 0] - center_xy[0], xy[:, 1] - center_xy[1])

    # --- Current land use (one-hot over the 11 MapPLUTO codes)
    for code in LAND_USE_LABELS:
        f[f"lu_{code}"] = (gdf["land_use"] == code).astype(float)

    return f.astype(float)


def transform_features(features_raw: pd.DataFrame) -> pd.DataFrame:
    """log1p heavy-tailed features, drop constant columns, z-score the rest."""
    f = features_raw.copy()
    for c in f.columns:
        if c in LOG_FEATURES:
            f[c] = np.log1p(f[c].clip(lower=0))
    std = f.std(ddof=0)
    keep = std[std > 1e-9].index
    dropped = sorted(set(f.columns) - set(keep))
    if dropped:
        logger.info("Dropping %d constant features: %s", len(dropped), dropped)
    f = f[keep]
    f = (f - f.mean()) / f.std(ddof=0)
    return f.fillna(0.0)


def _read_layer(path: Union[str, Path], where: Optional[str] = None,
                preferred_layer: str = "MapPLUTO") -> gpd.GeoDataFrame:
    """Read a parcel layer; ``where`` is an OGR SQL filter applied while reading.

    MapPLUTO archives contain both the shoreline-clipped ``MapPLUTO`` layer and
    ``MapPLUTO_UNCLIPPED``; the clipped layer is used when present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Parcel file not found: {path}. Download MapPLUTO from NYC DCP and pass --pluto PATH."
        )
    if path.suffix.lower() == ".parquet":
        g = gpd.read_parquet(path)
        return g
    src = f"zip://{path}" if path.suffix.lower() == ".zip" else str(path)
    kwargs = {}
    if where:
        kwargs["where"] = where
    try:
        import pyogrio

        layers = [lyr[0] for lyr in pyogrio.list_layers(src)]
        if len(layers) > 1:
            kwargs["layer"] = preferred_layer if preferred_layer in layers else layers[0]
    except Exception:  # noqa: BLE001 - fall back to the default layer
        pass
    return gpd.read_file(src, **kwargs)


class ParcelFileLoader:
    """Load a local parcel layer using a city YAML column mapping."""

    def __init__(self, city: str, parcel_path: Union[str, Path], cache_dir: Optional[Path] = None,
                 include_affordable_far: bool = False):
        self.city = city
        self.config: CityConfig = get_city_config(city)
        self.parcel_path = Path(parcel_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.include_affordable_far = include_affordable_far

    def _filter_study_area(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        boro = getattr(self.config, "borough_code", None)
        if boro is None:
            return gdf
        for col in ["BoroCode", "borocode"]:
            if col in gdf.columns:
                return gdf[pd.to_numeric(gdf[col], errors="coerce") == int(boro)]
        if "Borough" in gdf.columns:
            return gdf[gdf["Borough"].astype(str).str.upper().isin(["MN", "MANHATTAN"])]
        return gdf

    def load(self) -> ParcelDataset:
        boro = getattr(self.config, "borough_code", None)
        where = f"BoroCode = {int(boro)}" if boro is not None else None
        try:
            raw = _read_layer(self.parcel_path, where=where)
        except Exception:  # noqa: BLE001 - layer without BoroCode: filter after reading
            raw = _read_layer(self.parcel_path)
        raw = self._filter_study_area(raw)
        n_raw = len(raw)
        versions = sorted(raw["Version"].dropna().astype(str).unique()) if "Version" in raw else []
        gdf = standardise_parcels(raw, getattr(self.config, "column_mapping", {}), self.config.crs,
                                  include_affordable_far=self.include_affordable_far)

        n_before = len(gdf)
        gdf = gdf[(gdf["lot_area_sqft"] > 0)].reset_index(drop=True)
        center_xy = None
        if self.config.center_lon is not None:
            pt = gpd.GeoSeries(
                gpd.points_from_xy([self.config.center_lon], [self.config.center_lat]), crs="EPSG:4326"
            ).to_crs(self.config.crs)
            center_xy = np.array([pt.x.iloc[0], pt.y.iloc[0]])
        raw_f = compute_node_features(gdf, center_xy)
        feats = transform_features(raw_f)
        meta = {
            "city": self.city,
            "source_file": str(self.parcel_path),
            "n_parcels_in_study_area": int(n_raw),
            "n_parcels_used": int(len(gdf)),
            "n_dropped_zero_area": int(n_before - len(gdf)),
            "n_features_engineered": int(raw_f.shape[1]),
            "n_features_used": int(feats.shape[1]),
            "features_used": list(feats.columns),
            "crs": self.config.crs,
            "pluto_versions": versions,
            "source_sha256": _sha256(self.parcel_path),
            "include_affordable_far": self.include_affordable_far,
            "synthetic": False,
        }
        return ParcelDataset(gdf, raw_f, feats, list(feats.columns), meta)


def _sha256(path: Path) -> Optional[str]:
    import hashlib

    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ManhattanDataLoader(ParcelFileLoader):
    """MapPLUTO loader restricted to Manhattan (BoroCode 1)."""

    def __init__(self, parcel_path: Union[str, Path], cache_dir: Optional[Path] = None,
                 include_affordable_far: bool = False):
        super().__init__("manhattan", parcel_path, cache_dir, include_affordable_far)


class SyntheticCityLoader:
    """Synthetic grid city with MapPLUTO-like attributes.

    **For tests and CI smoke runs only.** Blocks of rectangular lots separated by
    streets; land use, zoning and floor area are drawn from simple distributions.
    """

    def __init__(self, n_blocks_x: int = 6, n_blocks_y: int = 8, lots_per_block: int = 8, seed: int = 0):
        self.nx, self.ny, self.lpb, self.seed = n_blocks_x, n_blocks_y, lots_per_block, seed

    def load(self) -> ParcelDataset:
        rng = np.random.default_rng(self.seed)
        block_w, block_h, street = 800.0, 200.0, 60.0
        lot_w = block_w / (self.lpb // 2)
        rows = []
        avenues = ["FIRST AVENUE", "SECOND AVENUE", "THIRD AVENUE", "LEXINGTON AVENUE",
                   "PARK AVENUE", "MADISON AVENUE", "FIFTH AVENUE", "SIXTH AVENUE"]
        zones = ["R6", "R7A", "R8", "C1-9", "C4-5", "C6-4", "M1-5"]
        zone_far = {"R6": (2.43, 0, 4.8), "R7A": (4.0, 0, 4.0), "R8": (6.02, 0, 6.5),
                    "C1-9": (10.0, 2.0, 10.0), "C4-5": (3.4, 3.4, 3.4),
                    "C6-4": (10.0, 10.0, 10.0), "M1-5": (0, 5.0, 6.5)}
        for bx in range(self.nx):
            for by in range(self.ny):
                zone = zones[(bx + 2 * by) % len(zones)] if rng.random() > 0.2 else zones[rng.integers(len(zones))]
                x0 = bx * (block_w + street)
                y0 = by * (block_h + street)
                street_no = 10 + by
                for k in range(self.lpb):
                    side = k // (self.lpb // 2)
                    i = k % (self.lpb // 2)
                    geom = box(x0 + i * lot_w, y0 + side * block_h / 2,
                               x0 + (i + 1) * lot_w, y0 + (side + 1) * block_h / 2)
                    lot_area = geom.area
                    if zone.startswith("M"):
                        lu = rng.choice(["06", "05", "10", "11"], p=[0.5, 0.2, 0.2, 0.1])
                    elif zone.startswith("C"):
                        lu = rng.choice(["05", "04", "03", "08", "09"], p=[0.45, 0.3, 0.15, 0.05, 0.05])
                    else:
                        lu = rng.choice(["01", "02", "03", "04", "08", "09", "11"],
                                        p=[0.05, 0.3, 0.35, 0.15, 0.05, 0.05, 0.05])
                    rf, cf, ff = zone_far[zone]
                    mx = max(rf, cf, ff)
                    built = 0.0 if lu in {"09", "11", "10"} else float(np.clip(rng.gamma(2.0, mx / 3.0), 0.2, 1.4 * mx))
                    floors = 0 if built == 0 else int(np.clip(round(built / rng.uniform(0.5, 0.9)), 1, 60))
                    bldg = built * lot_area
                    units = int(bldg / 900) if lu in RESIDENTIAL_CODES else 0
                    ass_land = lot_area * rng.uniform(80, 400) * (1.5 if zone.startswith("C") else 1.0)
                    ass_tot = ass_land + bldg * rng.uniform(50, 250)
                    addr_no = 100 + 20 * i
                    address = (f"{addr_no} EAST {street_no} STREET" if side == 0
                               else f"{addr_no} {avenues[bx % len(avenues)]}")
                    rows.append(dict(
                        geometry=geom, BBL=1_000_000_000 + len(rows), Address=address,
                        LotArea=lot_area, BldgArea=bldg, NumFloors=floors,
                        YearBuilt=int(rng.integers(1890, 2020)), YearAlter1=0,
                        LandUse=lu, ZoneDist1=zone, SPDist1=None, SplitZone="N",
                        BuiltFAR=built, ResidFAR=rf, CommFAR=cf, FacilFAR=ff,
                        AffResFAR=round(1.2 * rf, 2), ManuFAR=5.0 if zone.startswith("M") else 0.0,
                        OtherArea=bldg if lu == "08" else 0, BCTCB2020=f"1{bx:06d}{by:04d}",
                        MIHOption1="Option 1" if (rf > 0 and (bx + by) % 3 == 0) else None,
                        AssessTot=ass_tot, AssessLand=ass_land,
                        UnitsRes=units, UnitsTotal=units + (1 if lu in {"05", "06"} else 0),
                        BldgClass="D4", HistDist=None, Landmark=None,
                        LotFront=lot_w, LotDepth=block_h / 2, BldgFront=lot_w * 0.9, BldgDepth=block_h * 0.4,
                        IrrLotCode="N", ResArea=bldg if lu in RESIDENTIAL_CODES else 0,
                        ComArea=bldg if lu in {"05", "08"} else 0, OfficeArea=bldg if lu == "05" else 0,
                        RetailArea=0,
                        GarageArea=0, StrgeArea=0, FactryArea=bldg if lu == "06" else 0,
                        PFIRM15_FL=1 if (bx == 0 and rng.random() < 0.5) else 0, FIRM07_FLA=0,
                    ))
        raw = gpd.GeoDataFrame(rows, crs="EPSG:2263")
        cfg = get_city_config("manhattan")
        gdf = standardise_parcels(raw, cfg.column_mapping, "EPSG:2263")
        center = np.array([gdf["x"].mean(), gdf["y"].mean()])
        raw_f = compute_node_features(gdf, center)
        feats = transform_features(raw_f)
        meta = {
            "city": "synthetic", "source_file": None, "n_parcels_in_study_area": len(gdf),
            "n_parcels_used": len(gdf), "n_dropped_zero_area": 0,
            "n_features_engineered": int(raw_f.shape[1]), "n_features_used": int(feats.shape[1]),
            "features_used": list(feats.columns), "crs": "EPSG:2263", "synthetic": True,
        }
        return ParcelDataset(gdf, raw_f, feats, list(feats.columns), meta)


def get_data_loader(city: str, parcel_path: Optional[Union[str, Path]] = None, **kwargs):
    """Factory: ``'synthetic'`` or any city with a YAML config plus a parcel file."""
    if city == "synthetic":
        return SyntheticCityLoader(**kwargs)
    if parcel_path is None:
        raise ValueError(f"A local parcel file is required for city '{city}' (use --pluto PATH).")
    if city == "manhattan":
        return ManhattanDataLoader(parcel_path, **kwargs)
    return ParcelFileLoader(city, parcel_path, **kwargs)
