"""
Build lot-level city context from the downloaded sources (see :mod:`.fetch`).

Walking network
    NYC Street Centerline segments of walkable types (streets, boardwalks,
    paths, step streets, alleys; not highways, ramps, tunnels, driveways or
    ferries) form an undirected graph whose nodes are segment end points
    (EPSG:2263, feet). Lots and destinations attach to their nearest node.
    A 15-minute walk is ``walk_minutes`` x ``walk_speed_m_per_min`` metres of
    network distance (default 15 x 80 m = 1,200 m).

Everyday destinations (six categories after Moreno et al., 2021)
    food       retail food stores (NYS Agriculture and Markets licences)
    health     hospitals and clinics (DCP Facilities Database)
    education  K-12 schools and day care / pre-K (Facilities Database)
    parks      NYC Parks properties (entrances approximated by boundary points)
    civic      public libraries, community centres and cultural institutions
    transit    subway stations (MTA)

Jobs and people
    Jobs (LODES 2023, workplace blocks) and employed residents (LODES 2023,
    residence blocks) are distributed to lots within each 2020 census block by
    non-residential and residential floor area. Residents come from the 2020
    Census block population (P.L. 94-171), distributed by residential units.
    Square feet per job by use is estimated from block jobs and PLUTO floor
    areas by non-negative least squares.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

CATEGORIES = ["food", "health", "education", "parks", "civic", "transit"]
SERVICE_CATEGORIES = ["health", "education", "civic"]  # what new community-facility space can provide
WALKABLE_RW_TYPES = {"1", "5", "6", "7", "10"}  # street, boardwalk, path/trail, step street, alley
USES = ["res", "office", "retail", "facility"]
FT_PER_M = 3.28084
MANHATTAN_COUNTY = "36061"


@dataclass
class CityContext:
    """Lot-aligned context arrays. ``reach`` is node x node (within the walk threshold)."""

    node_of_lot: np.ndarray
    reach: csr_matrix
    has0: np.ndarray            # nodes x K: category reachable under existing conditions
    minutes0: np.ndarray        # nodes x K: walking minutes to nearest destination
    gap_service: np.ndarray     # nodes: index into CATEGORIES of the least-served service category
    pop0: np.ndarray            # lots: residents (2020)
    workers0: np.ndarray        # lots: employed residents (LODES RAC 2023)
    jobs0: np.ndarray           # lots: jobs (LODES WAC 2023)
    area0: Dict[str, np.ndarray]  # lots: existing floor area by use (sq ft)
    params: Dict[str, float]    # calibrated quantities
    meta: Dict = field(default_factory=dict)
    supply0: np.ndarray = None  # nodes x K: destination supply (see _destinations)

    @property
    def n_nodes(self) -> int:
        return self.reach.shape[0]


# --------------------------------------------------------------------------- network
def _walk_network(streets_path: Path):
    import geopandas as gpd

    s = gpd.read_file(streets_path)
    s = s[s["rw_type"].astype(str).isin(WALKABLE_RW_TYPES)].to_crs(2263)
    starts, ends, lengths = [], [], []
    for geom in s.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for ln in parts:
            c = np.asarray(ln.coords)
            starts.append(c[0][:2])
            ends.append(c[-1][:2])
            lengths.append(ln.length)
    starts, ends = np.round(np.array(starts)), np.round(np.array(ends))
    pts = np.vstack([starts, ends])
    uniq, inv = np.unique(pts, axis=0, return_inverse=True)
    m = len(starts)
    a, b = inv[:m], inv[m:]
    w = np.maximum(np.array(lengths), 1.0)
    keep = a != b
    n = len(uniq)
    g = coo_matrix((np.r_[w[keep], w[keep]], (np.r_[a[keep], b[keep]], np.r_[b[keep], a[keep]])), shape=(n, n)).tocsr()
    # Keep the largest connected component (Manhattan's main walking network).
    from scipy.sparse.csgraph import connected_components

    ncomp, lab = connected_components(g, directed=False)
    main = np.bincount(lab).argmax()
    idx = np.where(lab == main)[0]
    g = g[idx][:, idx]
    return uniq[idx], g


def _reach(g: csr_matrix, limit_ft: float, batch: int = 512) -> csr_matrix:
    n = g.shape[0]
    rows, cols = [], []
    for s in range(0, n, batch):
        src = np.arange(s, min(n, s + batch))
        d = dijkstra(g, directed=False, indices=src, limit=limit_ft)
        r, c = np.nonzero(np.isfinite(d))
        rows.append(src[r])
        cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    return csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n))


def _to_2263(lon, lat):
    from pyproj import Transformer

    t = Transformer.from_crs(4326, 2263, always_xy=True)
    return np.column_stack(t.transform(np.asarray(lon, float), np.asarray(lat, float)))


# ----------------------------------------------------------------------- destinations
def _destinations(ctx_dir: Path) -> Dict[str, tuple]:
    """Destination points (EPSG:2263) and supply weights per category.

    Supply: food = store floor area (sq ft; median where missing), parks = acres
    (split over boundary access points), education = enrolment capacity where
    reported (median otherwise), health / civic / transit = 1 per site.
    """
    import geopandas as gpd

    fac = pd.read_csv(ctx_dir / "facilities" / "facilities.csv", dtype=str)
    fac = fac.dropna(subset=["latitude", "longitude"])
    xy = _to_2263(fac["longitude"].astype(float), fac["latitude"].astype(float))
    grp, sub = fac["facgroup"].fillna(""), fac["facsubgrp"].fillna("")
    sel = {
        "health": sub.eq("HOSPITALS AND CLINICS"),
        "education": grp.eq("SCHOOLS (K-12)") | grp.eq("DAY CARE AND PRE-KINDERGARTEN"),
        "civic": sub.isin(["PUBLIC LIBRARIES", "COMMUNITY CENTERS AND COMMUNITY PROGRAMS",
                           "MUSEUMS", "OTHER CULTURAL INSTITUTIONS"]),
    }
    out = {k: (xy[v.values], np.ones(int(v.sum()))) for k, v in sel.items()}
    cap = pd.to_numeric(fac["capacity"], errors="coerce").values
    ed = sel["education"].values
    ed_cap = cap[ed]
    med = np.nanmedian(ed_cap[ed_cap > 0]) if np.any(ed_cap > 0) else 1.0
    out["education"] = (xy[ed], np.where(ed_cap > 0, ed_cap, med))

    food = pd.read_csv(ctx_dir / "food_stores" / "food_stores.csv", dtype=str)
    pt = food["georeference"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    sq = pd.to_numeric(food["square_footage"], errors="coerce")
    ok = pt.notna().all(1).values
    sq = sq[ok]
    out["food"] = (_to_2263(pt[0][ok], pt[1][ok]), sq.fillna(sq[sq > 0].median()).clip(lower=100).values)

    sub_ = pd.read_csv(ctx_dir / "subway" / "subway_stations.csv")
    out["transit"] = (_to_2263(sub_["gtfs_longitude"], sub_["gtfs_latitude"]), np.ones(len(sub_)))

    parks = gpd.read_file(ctx_dir / "parks" / "parks.geojson").to_crs(2263)
    parks = parks[~parks["typecategory"].isin(["Buildings/Institutions", "Parkway", "Undeveloped"])]
    pts: List[np.ndarray] = []
    wts: List[np.ndarray] = []
    for geom, area_sqft in zip(parks.geometry, parks.geometry.area):
        if geom is None or geom.is_empty:
            continue
        b = geom.boundary
        n = max(1, int(b.length // 300))  # a boundary point every ~300 ft approximates entrances
        pts.append(np.array([[*b.interpolate(i / n, normalized=True).coords[0][:2]] for i in range(n)]))
        wts.append(np.full(n, area_sqft / 43560.0 / n))
    out["parks"] = (np.vstack(pts), np.concatenate(wts))
    return out


# ------------------------------------------------------------------------- carbon
CLF_USE = {"res": ["Residential: Multifamily (5 or more units)"], "office": ["Office"],
           "retail": ["Office", "Mercantile"], "facility": ["Education", "Healthcare", "Public Assembly"]}
LL84_USE = {"res": ["Multifamily Housing"], "office": ["Office"],
            "retail": ["Retail Store", "Supermarket/Grocery Store", "Other - Mall", "Strip Mall", "Food Sales"],
            "facility": ["K-12 School", "College/University", "Hospital (General Medical & Surgical)",
                         "Medical Office", "Library", "Museum", "Pre-school/Daycare", "Social/Meeting Hall",
                         "Worship Facility", "Other - Education", "Ambulatory Surgical Center",
                         "Outpatient Rehabilitation/Physical Therapy"]}


def carbon_factors(ctx_dir: Path) -> Dict[str, float]:
    """Embodied (CLF WBLCA v2, new construction, A-C, kg CO2e/m2 GFA, median) and
    operational (LL84 2024, Manhattan buildings built since 2010, kg CO2e/ft2/yr,
    median of location-based GHG / calculated GFA) carbon intensity by use, with
    interquartile ranges for the uncertainty analysis."""
    out: Dict[str, float] = {}
    clf = pd.read_excel(ctx_dir / "clf_wblca" / "buildings_metadata.xlsx")
    clf = clf[clf["bldg_proj_type"].astype(str).str.startswith("New")]
    for u, cats in CLF_USE.items():
        v = pd.to_numeric(clf.loc[clf["bldg_prim_use"].isin(cats), "eci_a_to_c_gfa"], errors="coerce").dropna()
        out[f"eci_{u}_kg_m2"], out[f"eci_{u}_q25"], out[f"eci_{u}_q75"] = map(float, v.quantile([0.5, 0.25, 0.75]))
        out[f"eci_{u}_n"] = int(len(v))
    ll = pd.read_csv(ctx_dir / "ll84" / "ll84_2024_manhattan_built2010plus.csv", low_memory=False)
    ghg = pd.to_numeric(ll["total_location_based_ghg"], errors="coerce")
    gfa = pd.to_numeric(ll["property_gfa_calculated"], errors="coerce")
    inten = 1000.0 * ghg / gfa  # kg CO2e per ft2 per year
    ok = (gfa > 5000) & inten.between(0.2, 60)
    for u, cats in LL84_USE.items():
        v = inten[ok & ll["primary_property_type"].isin(cats)].dropna()
        out[f"opc_{u}_kg_ft2"], out[f"opc_{u}_q25"], out[f"opc_{u}_q75"] = map(float, v.quantile([0.5, 0.25, 0.75]))
        out[f"opc_{u}_n"] = int(len(v))
    return out


# --------------------------------------------------------------------- jobs & people
def _pl_block_population(zip_path: Path) -> pd.Series:
    """2020 P.L. 94-171 total population (P1_001N) for Manhattan blocks, by GEOID."""
    z = zipfile.ZipFile(zip_path)
    geo = {}
    with z.open("nygeo2020.pl") as f:
        for line in io.TextIOWrapper(f, encoding="latin-1"):
            p = line.split("|")
            if p[2] == "750" and p[14] == "061":  # SUMLEV block, county New York
                geo[p[7]] = p[9]  # LOGRECNO -> GEOCODE (15-digit block GEOID)
    pop = {}
    with z.open("ny000012020.pl") as f:
        for line in io.TextIOWrapper(f, encoding="latin-1"):
            p = line.split("|", 6)
            g = geo.get(p[4])
            if g is not None:
                pop[g] = int(p[5])
    return pd.Series(pop, name="pop")


def _lodes(path: Path, col: str = "C000") -> pd.Series:
    with gzip.open(path, "rt") as f:
        d = pd.read_csv(f, dtype={"w_geocode": str, "h_geocode": str})
    key = "w_geocode" if "w_geocode" in d else "h_geocode"
    d = d[d[key].str.startswith(MANHATTAN_COUNTY)]
    return d.set_index(key)[col].astype(float)


def _spread(block_total: pd.Series, lot_block: pd.Series, weight: np.ndarray, block_xy: Dict[str, np.ndarray],
            lot_xy: np.ndarray) -> np.ndarray:
    """Distribute block totals to the block's lots by ``weight``; blocks without
    weighted lots go to the nearest lot of the block point (or nearest lot)."""
    n = len(lot_block)
    out = np.zeros(n)
    df = pd.DataFrame({"b": lot_block.values, "w": weight, "i": np.arange(n)})
    wsum = df.groupby("b")["w"].sum()
    tot = block_total.reindex(wsum.index).fillna(0.0)
    has = wsum[wsum > 0].index
    m = df["b"].isin(has)
    share = df.loc[m, "w"] / wsum.loc[df.loc[m, "b"]].values
    out[df.loc[m, "i"].values] = share.values * tot.loc[df.loc[m, "b"]].values
    rest = block_total[~block_total.index.isin(has)]
    rest = rest[rest > 0]
    if len(rest):
        tree = cKDTree(lot_xy)
        xy = np.array([block_xy.get(b, (np.nan, np.nan)) for b in rest.index])
        ok = np.isfinite(xy).all(1)
        _, j = tree.query(xy[ok])
        np.add.at(out, j, rest.values[ok])
    return out


def build_context(lots: pd.DataFrame, ctx_dir: str | Path, walk_minutes: float = 15.0,
                  walk_speed_m_per_min: float = 80.0) -> CityContext:
    """``lots`` needs x, y (EPSG:2263), bctcb2020, units_res and floor areas
    res_area, office_area, retail_area, com_area, other_area, land_use."""
    ctx_dir = Path(ctx_dir)
    limit_ft = walk_minutes * walk_speed_m_per_min * FT_PER_M
    feet_per_min = walk_speed_m_per_min * FT_PER_M
    lot_xy = lots[["x", "y"]].to_numpy(float)

    nodes, g = _walk_network(ctx_dir / "streets" / "streets.geojson")
    tree = cKDTree(nodes)
    node_of_lot = tree.query(lot_xy)[1]
    logger.info("walk network: %d nodes, %d edges", len(nodes), g.nnz // 2)
    reach = _reach(g, limit_ft)
    logger.info("walksheds: %.0f nodes reachable on average", reach.nnz / reach.shape[0])

    dests = _destinations(ctx_dir)
    K = len(CATEGORIES)
    minutes0 = np.zeros((len(nodes), K))
    counts = np.zeros((len(nodes), K))
    supply0 = np.zeros((len(nodes), K))
    for k, c in enumerate(CATEGORIES):
        pts, wts = dests[c]
        dn_all = tree.query(pts)[1]
        d = dijkstra(g, directed=False, indices=np.unique(dn_all), min_only=True)
        minutes0[:, k] = d / feet_per_min
        ind = np.zeros(len(nodes))
        np.add.at(ind, dn_all, 1.0)
        np.add.at(supply0[:, k], dn_all, wts)
        counts[:, k] = reach @ ind
    has0 = minutes0 <= walk_minutes
    svc = [CATEGORIES.index(c) for c in SERVICE_CATEGORIES]
    gap_service = np.array(svc)[np.argmin(counts[:, svc], axis=1)]

    # Floor area by use.
    lu = lots["land_use"].astype(str).str.zfill(2).values
    other = lots["other_area"].fillna(0).to_numpy(float)
    area0 = {
        "res": lots["res_area"].fillna(0).to_numpy(float),
        "office": lots["office_area"].fillna(0).to_numpy(float),
        "retail": lots["retail_area"].fillna(0).to_numpy(float),
        "facility": np.where(lu == "08", other, 0.0),
    }
    com = lots["com_area"].fillna(0).to_numpy(float)
    nonres_other = np.maximum(com - area0["office"] - area0["retail"] - area0["facility"], 0.0)

    # Jobs, workers and people by block.
    xw = pd.read_csv(ctx_dir / "lodes_xwalk" / "ny_xwalk.csv.gz", dtype={"tabblk2020": str},
                     usecols=["tabblk2020", "blklatdd", "blklondd"])
    xw = xw[xw["tabblk2020"].str.startswith(MANHATTAN_COUNTY)]
    bxy = dict(zip(xw["tabblk2020"], map(tuple, _to_2263(xw["blklondd"], xw["blklatdd"]))))
    bct = lots["bctcb2020"].astype(str).str.zfill(11)
    geoid = MANHATTAN_COUNTY + bct.str[1:]
    jobs_b = _lodes(ctx_dir / "lodes_wac" / "ny_wac_S000_JT00_2023.csv.gz")
    work_b = _lodes(ctx_dir / "lodes_rac" / "ny_rac_S000_JT00_2023.csv.gz")
    pop_b = _pl_block_population(ctx_dir / "census_pl" / "ny2020.pl.zip")
    nonres = area0["office"] + area0["retail"] + area0["facility"] + nonres_other
    units = lots["units_res"].fillna(0).to_numpy(float)
    jobs0 = _spread(jobs_b, geoid, nonres, bxy, lot_xy)
    workers0 = _spread(work_b, geoid, np.maximum(area0["res"], units), bxy, lot_xy)
    pop0 = _spread(pop_b, geoid, np.where(units > 0, units, area0["res"] / 1000.0), bxy, lot_xy)

    # Square feet per job by use: block jobs ~ office/a + retail/b + facility/c + other/d (NNLS).
    blk = pd.DataFrame({"b": geoid.values, "office": area0["office"], "retail": area0["retail"],
                        "facility": area0["facility"], "other": nonres_other}).groupby("b").sum()
    blk["jobs"] = jobs_b.reindex(blk.index).fillna(0.0)
    blk = blk[(blk[["office", "retail", "facility", "other"]].sum(1) > 0)]
    X = blk[["office", "retail", "facility", "other"]].to_numpy() / 1000.0
    coef, _ = nnls(X, blk["jobs"].to_numpy())
    jobs_per_ksf = dict(zip(["office", "retail", "facility", "other"], coef))
    pred = X @ coef
    r2 = 1 - ((blk["jobs"] - pred) ** 2).sum() / ((blk["jobs"] - blk["jobs"].mean()) ** 2).sum()

    res_lots = units > 0
    params = {
        "walk_limit_ft": limit_ft, "walk_minutes": walk_minutes,
        "persons_per_unit": float(pop0[res_lots].sum() / units[res_lots].sum()),
        "employed_per_resident": float(workers0.sum() / max(pop0.sum(), 1.0)),
        "jobs_per_ksf_office": jobs_per_ksf["office"], "jobs_per_ksf_retail": jobs_per_ksf["retail"],
        "jobs_per_ksf_facility": jobs_per_ksf["facility"], "jobs_per_ksf_other": jobs_per_ksf["other"],
        "jobs_model_r2": float(r2),
    }
    params.update(carbon_factors(ctx_dir))
    params["unit_sqft_new"] = _new_unit_size(lots)
    meta = {
        "n_nodes": int(len(nodes)), "mean_walkshed_nodes": float(reach.nnz / reach.shape[0]),
        "n_destinations": {c: int(len(dests[c][0])) for c in CATEGORIES},
        "supply_totals": {c: float(supply0[:, k].sum()) for k, c in enumerate(CATEGORIES)},
        "jobs_total": float(jobs_b.sum()), "workers_total": float(work_b.sum()), "pop_total": float(pop_b.sum()),
        "jobs_assigned": float(jobs0.sum()), "pop_assigned": float(pop0.sum()),
        "share_residents_all_categories": float((pop0 * has0[node_of_lot].all(1)).sum() / pop0.sum()),
    }
    logger.info("context: %s", json.dumps(meta))
    return CityContext(node_of_lot=node_of_lot, reach=reach, has0=has0, minutes0=minutes0,
                       gap_service=gap_service, pop0=pop0, workers0=workers0, jobs0=jobs0,
                       area0=area0, params=params, meta=meta, supply0=supply0)


def _new_unit_size(lots: pd.DataFrame) -> float:
    """Median residential floor area per unit in buildings completed since 2010 with 10+ units."""
    yb = pd.to_numeric(lots.get("year_built"), errors="coerce")
    u = lots["units_res"].fillna(0)
    m = (yb >= 2010) & (u >= 10) & (lots["res_area"].fillna(0) > 0)
    return float((lots.loc[m, "res_area"] / u[m]).median()) if m.any() else 1000.0


def save_context(ctx: CityContext, path: str | Path) -> None:
    from scipy.sparse import save_npz

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_npz(path / "reach.npz", ctx.reach)
    np.savez_compressed(path / "arrays.npz", node_of_lot=ctx.node_of_lot, has0=ctx.has0, minutes0=ctx.minutes0,
                        gap_service=ctx.gap_service, pop0=ctx.pop0, workers0=ctx.workers0, jobs0=ctx.jobs0,
                        supply0=ctx.supply0, **{f"area0_{k}": v for k, v in ctx.area0.items()})
    (path / "meta.json").write_text(json.dumps({"params": ctx.params, "meta": ctx.meta}, indent=1))


def load_context(path: str | Path) -> CityContext:
    from scipy.sparse import load_npz

    path = Path(path)
    a = np.load(path / "arrays.npz")
    m = json.loads((path / "meta.json").read_text())
    return CityContext(node_of_lot=a["node_of_lot"], reach=load_npz(path / "reach.npz").tocsr(), has0=a["has0"],
                       minutes0=a["minutes0"], gap_service=a["gap_service"], pop0=a["pop0"],
                       workers0=a["workers0"], jobs0=a["jobs0"],
                       area0={k: a[f"area0_{k}"] for k in USES}, params=m["params"], meta=m["meta"],
                       supply0=a["supply0"])


def synthetic_context(lots: pd.DataFrame, seed: int = 0, walk_limit_ft: float = 1500.0) -> CityContext:
    """Context for the synthetic test city (tests and CI only): every lot is a
    network node, walksheds are Euclidean distances x 1.2 within ``walk_limit_ft``,
    destinations are random lots, people and jobs follow floor area, and the
    calibrated parameters take Manhattan-like default values."""
    rng = np.random.default_rng(seed)
    xy = lots[["x", "y"]].to_numpy(float)
    n = len(xy)
    pairs = cKDTree(xy).sparse_distance_matrix(cKDTree(xy), walk_limit_ft / 1.2, output_type="coo_matrix")
    reach = csr_matrix((np.ones(pairs.nnz, dtype=np.float32), (pairs.row, pairs.col)), shape=(n, n))
    reach = (reach + csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), np.arange(n))), shape=(n, n)))
    reach.data[:] = 1.0
    K = len(CATEGORIES)
    supply0 = np.zeros((n, K))
    for k in range(K):
        idx = rng.choice(n, size=max(2, n // 25), replace=False)
        supply0[idx, k] = rng.uniform(1, 10, len(idx))
    has0 = (reach @ (supply0 > 0).astype(float)) > 0
    minutes0 = np.where(has0, 5.0, 25.0)
    svc = [CATEGORIES.index(c) for c in SERVICE_CATEGORIES]
    gap_service = np.array(svc)[np.argmin((reach @ supply0)[:, svc], axis=1)]
    lu = lots["land_use"].astype(str).str.zfill(2).values
    area0 = {"res": lots["res_area"].fillna(0).to_numpy(float),
             "office": lots["office_area"].fillna(0).to_numpy(float),
             "retail": lots["retail_area"].fillna(0).to_numpy(float),
             "facility": np.where(lu == "08", lots.get("other_area", pd.Series(0, index=lots.index)).fillna(0), 0.0)}
    units = lots["units_res"].fillna(0).to_numpy(float)
    pop0 = units * 1.8
    params = {"walk_limit_ft": walk_limit_ft, "walk_minutes": 15.0, "persons_per_unit": 1.8,
              "employed_per_resident": 0.46, "jobs_per_ksf_office": 2.9, "jobs_per_ksf_retail": 3.5,
              "jobs_per_ksf_facility": 0.95, "jobs_per_ksf_other": 0.85, "jobs_model_r2": float("nan"),
              "unit_sqft_new": 1000.0}
    for u, (e, o) in {"res": (404, 4.6), "office": (571, 6.0), "retail": (575, 10.1), "facility": (548, 6.8)}.items():
        params.update({f"eci_{u}_kg_m2": e, f"eci_{u}_q25": 0.8 * e, f"eci_{u}_q75": 1.25 * e,
                       f"opc_{u}_kg_ft2": o, f"opc_{u}_q25": 0.8 * o, f"opc_{u}_q75": 1.25 * o})
    jobs0 = (area0["office"] * 2.9 + area0["retail"] * 3.5 + area0["facility"] * 0.95) / 1000.0
    return CityContext(node_of_lot=np.arange(n), reach=reach.tocsr(), has0=has0, minutes0=minutes0,
                       gap_service=gap_service, pop0=pop0, workers0=pop0 * 0.46, jobs0=jobs0, area0=area0,
                       params=params, meta={"synthetic": True}, supply0=supply0)
