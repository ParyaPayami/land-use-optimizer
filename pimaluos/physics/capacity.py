"""
Planning-scale capacity models (the "Verify" layer).

These are deliberately simple, vectorised screening models that run in O(N)
(or O(N k) for the shadow screen) so they can be evaluated at every MARL step on
all Manhattan lots. They are **not** traffic assignment, hydraulic sewer models
or ray-traced shadow studies, and the manuscript describes them as screens.

Building envelope
    A plan sets a floor-area ratio (FAR) per lot. Footprint coverage is kept at
    its existing value unless the new FAR cannot fit within the existing number
    of floors; coverage then grows up to ``max_coverage`` and any remaining floor
    area is added as height. Height = floors x ``floor_height_ft``.

Traffic (BPR screen)
    Peak-hour vehicle trips per lot = floor area by land-use class x trip rate.
    Lots are aggregated to square traffic-analysis cells (``taz_size_ft``).
    Cell capacity is proportional to the street frontage of its lots, with one
    global constant calibrated so that the median existing-conditions V/C equals
    ``vc_reference``; it is floored at the capacity at which the cell's existing
    demand runs at ``vc_reference`` (the existing network is assumed to serve
    existing trips). Travel-time index = 1 + alpha (V/C)^beta. A cell is over
    capacity when V/C > ``vc_threshold``.

Hydrology (Rational Method screen)
    Peak combined-sewer load per catchment cell (``catchment_size_ft``) =
    sum_i C_i I A_i (cfs) + peaking x sanitary flow (proportional to floor area).
    The existing system is assumed to carry existing load plus ``sewer_headroom``;
    a catchment is over capacity when the plan's load exceeds that.

Solar access (winter-solstice noon screen)
    Shadow length L = H / tan(altitude). Lot j is shaded by lot i if j lies north
    of i within L, overlaps i's east-west extent, and H_i > H_j. Candidate pairs
    are the ``shadow_k`` nearest lots. Reported: lots newly shaded relative to
    existing conditions.

Equity proxies (MapPLUTO only; no census join)
    * Green space per resident: open-space lot area in the 3x3 cell
      neighbourhood divided by residents (residential floor area x
      ``persons_per_1000sqft``); Gini over residential lots weighted by residents.
    * Displacement exposure: share of added floor area that falls on
      residential lots in the bottom quartile of assessed value per unit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pimaluos.core.data_loader import RESIDENTIAL_CODES

DEFAULT_TRIP_RATES = {  # vehicle trips per 1,000 sq ft, peak hour (urban, low auto share)
    "residential": 0.15, "mixed": 0.30, "commercial": 0.50, "industrial": 0.30,
    "public": 0.30, "utility": 0.10, "parking": 0.0, "open_space": 0.0, "vacant": 0.0,
}
DEFAULT_PERVIOUS_C = {  # runoff coefficient of the unbuilt part of the lot
    "open_space": 0.20, "vacant": 0.40, "parking": 0.90,
}


@dataclass
class CapacityParams:
    floor_height_ft: float = 11.0
    max_coverage: float = 0.80
    taz_size_ft: float = 1320.0
    catchment_size_ft: float = 2640.0
    vc_reference: float = 0.70
    vc_threshold: float = 1.00
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0
    rainfall_in_per_hr: float = 1.75
    roof_c: float = 0.95
    paved_c: float = 0.85
    sanitary_cfs_per_1000sqft: float = 0.0002
    sanitary_peaking: float = 2.5
    sewer_headroom: float = 0.10
    latitude_deg: float = 40.78
    shadow_k: int = 24
    green_radius_cells: int = 1
    persons_per_1000sqft: float = 2.0
    trip_rates: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TRIP_RATES))

    @property
    def sun_altitude_deg(self) -> float:
        return 90.0 - self.latitude_deg - 23.44


def gini(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Weighted Gini coefficient (0 = equal)."""
    v = np.asarray(values, dtype=float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)
    m = (w > 0) & np.isfinite(v)
    v, w = v[m], w[m]
    if len(v) == 0 or v.sum() <= 0:
        return 0.0
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cvw = np.cumsum(v * w)
    total_w, total_vw = cw[-1], cvw[-1]
    # Area under the Lorenz curve (trapezoids).
    lorenz = cvw / total_vw
    prev = np.concatenate([[0.0], lorenz[:-1]])
    area = np.sum((w / total_w) * (lorenz + prev) / 2.0)
    return float(1.0 - 2.0 * area)


class CapacityModel:
    """Vectorised evaluation of a FAR plan against the capacity screens."""

    def __init__(self, gdf: pd.DataFrame, params: Optional[CapacityParams] = None):
        self.p = params or CapacityParams()
        p = self.p
        g = gdf.reset_index(drop=True)
        self.n = len(g)
        self.A = g["lot_area_sqft"].values.astype(float)
        self.far0 = g["built_far"].values.astype(float)
        self.max_far = g["max_far"].values.astype(float)
        # Upper bound for plans: zoning maximum, but existing (legally
        # non-complying) bulk above the maximum is retained, never forced down.
        self.ub = np.maximum(self.max_far, self.far0)
        self.floors0 = g["num_floors"].values.astype(float)
        self.lu = g["land_use"].astype(str).values
        self.cls = g["land_use_class"].astype(str).values
        x, y = g["x"].values.astype(float), g["y"].values.astype(float)
        self.x, self.y = x, y

        bldg = self.far0 * self.A
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = np.where(self.floors0 > 0, bldg / np.maximum(self.floors0, 1) / self.A, 0.0)
        self.cov0 = np.clip(np.nan_to_num(cov), 0, 1)
        self.cov_cap = np.maximum(self.cov0, p.max_coverage)

        # Residential share of floor area.
        res_area = g["res_area"].fillna(0).values if "res_area" in g else np.zeros(self.n)
        share = np.where(np.isin(self.lu, ["01", "02", "03"]), 1.0, 0.0)
        mixed = self.lu == "04"
        with np.errstate(divide="ignore", invalid="ignore"):
            mixed_share = np.where(bldg > 0, np.clip(res_area / bldg, 0, 1), 0.6)
        share = np.where(mixed, np.where(mixed_share > 0, mixed_share, 0.6), share)
        self.res_share = share
        self.is_res = np.isin(self.lu, list(RESIDENTIAL_CODES))
        self.flood = (g["flood_zone"].values > 0) if "flood_zone" in g else np.zeros(self.n, bool)
        if "flood_2015" in g or "flood_2007" in g:
            f15 = pd.to_numeric(g.get("flood_2015", 0), errors="coerce").fillna(0).values > 0
            f07 = pd.to_numeric(g.get("flood_2007", 0), errors="coerce").fillna(0).values > 0
            self.flood = f15 | f07

        # Grid cells.
        x0, y0 = x.min(), y.min()
        self.taz = self._cell_ids(x, y, x0, y0, p.taz_size_ft)
        self.catch = self._cell_ids(x, y, x0, y0, p.catchment_size_ft)
        self.n_taz = int(self.taz.max()) + 1
        self.n_catch = int(self.catch.max()) + 1

        # Traffic capacity from frontage.
        front = g["lot_front"].fillna(0).values if "lot_front" in g else np.zeros(self.n)
        front = np.where(front > 0, front, np.sqrt(self.A))
        self.rate = np.array([p.trip_rates.get(c, 0.0) for c in self.cls]) / 1000.0
        frontage_taz = np.bincount(self.taz, weights=front, minlength=self.n_taz)
        demand0 = self._demand(self.far0)
        valid = (frontage_taz > 0) & (demand0 > 0)
        raw_vc = np.where(valid, demand0 / np.maximum(frontage_taz, 1e-9), 0.0)
        kappa = np.median(raw_vc[valid]) / p.vc_reference if valid.any() else 1.0
        self.taz_over_frontage_only = int((demand0 / np.maximum(frontage_taz * kappa, 1e-9) > p.vc_threshold).sum())
        # A cell's capacity is at least what its existing demand needs to run at the
        # reference V/C: the existing network is assumed to serve existing trips
        # (Manhattan's avenues and transit are not captured by lot frontage).
        self.cap_taz = np.maximum(np.maximum(frontage_taz * kappa, demand0 / p.vc_reference), 1e-9)

        # Hydrology capacity = existing load x (1 + headroom).
        self.c_perv = np.array([DEFAULT_PERVIOUS_C.get(c, p.paved_c) for c in self.cls])
        self.load0_catch = self._sewer_load(self.far0)
        self.cap_catch = self.load0_catch * (1.0 + p.sewer_headroom)

        # Shadow candidate pairs (i casts on j).
        k = int(min(p.shadow_k + 1, self.n))
        _, idx = cKDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=k)
        idx = idx[:, 1:] if k > 1 else np.zeros((self.n, 0), int)
        ii = np.repeat(np.arange(self.n), idx.shape[1])
        jj = idx.ravel()
        dy = y[jj] - y[ii]
        half_w = 0.5 * np.sqrt(self.A)
        lateral = np.abs(x[jj] - x[ii]) <= (half_w[ii] + half_w[jj])
        keep = (dy > 0) & lateral
        self.pair_i, self.pair_j, self.pair_dy = ii[keep], jj[keep], dy[keep]
        self.tan_alt = np.tan(np.radians(p.sun_altitude_deg))

        # Equity: open space within the 3x3 cell neighbourhood.
        self._nbr = self._cell_neighbourhood(x, y, x0, y0, p.taz_size_ft, p.green_radius_cells)
        open_area = np.where(self.lu == "09", self.A, 0.0)
        self.green_area_nbr = self._nbr @ np.bincount(self.taz, weights=open_area, minlength=self.n_taz)
        # Displacement vulnerability: bottom quartile of assessed value per unit.
        units = g["units_res"].fillna(0).values if "units_res" in g else np.zeros(self.n)
        val = g["assessed_total"].fillna(0).values if "assessed_total" in g else np.zeros(self.n)
        vpu = np.where(units > 0, val / np.maximum(units, 1), np.nan)
        res_units = self.is_res & np.isfinite(vpu)
        q = np.nanquantile(vpu[res_units], 0.25) if res_units.any() else np.inf
        self.vulnerable = res_units & (vpu <= q)
        self.value_per_bldg_sqft = np.where(bldg > 0, val / np.maximum(bldg, 1), np.nan)
        med = np.nanmedian(self.value_per_bldg_sqft) if np.isfinite(self.value_per_bldg_sqft).any() else 1.0
        self.value_per_bldg_sqft = np.nan_to_num(self.value_per_bldg_sqft, nan=med)

        self.baseline = self.evaluate(self.far0)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _cell_ids(x, y, x0, y0, size):
        cx = np.floor((x - x0) / size).astype(np.int64)
        cy = np.floor((y - y0) / size).astype(np.int64)
        ncx = int(cx.max()) + 1
        ids = cy * ncx + cx
        _, inv = np.unique(ids, return_inverse=True)
        return inv

    def _cell_neighbourhood(self, x, y, x0, y0, size, r):
        """Sparse (n_taz x n_taz) matrix linking each cell to cells within r rings."""
        from scipy.sparse import csr_matrix

        cx = np.floor((x - x0) / size).astype(np.int64)
        cy = np.floor((y - y0) / size).astype(np.int64)
        cells = pd.DataFrame({"t": self.taz, "cx": cx, "cy": cy}).drop_duplicates("t").set_index("t")
        lookup = {(a, b): t for t, (a, b) in cells[["cx", "cy"]].iterrows()}
        rows, cols = [], []
        for t, (a, b) in cells[["cx", "cy"]].iterrows():
            for da in range(-r, r + 1):
                for db in range(-r, r + 1):
                    u = lookup.get((a + da, b + db))
                    if u is not None:
                        rows.append(t)
                        cols.append(u)
        return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n_taz, self.n_taz))

    def envelope(self, far: np.ndarray):
        """Return (coverage, floors, height_ft) for a FAR vector."""
        far = np.maximum(far, 0.0)
        floors_existing = np.maximum(self.floors0, 1.0)
        needed_cov = far / floors_existing
        cov = np.where(far <= self.far0, self.cov0, np.clip(np.maximum(self.cov0, needed_cov), 0, self.cov_cap))
        cov = np.where((cov <= 0) & (far > 0), np.minimum(self.p.max_coverage, far), cov)
        with np.errstate(divide="ignore", invalid="ignore"):
            floors = np.where(cov > 0, far / cov, 0.0)
        floors = np.where(far > 0, np.maximum(floors, 1.0), 0.0)
        return cov, floors, floors * self.p.floor_height_ft

    def _demand(self, far):
        return np.bincount(self.taz, weights=far * self.A * self.rate, minlength=self.n_taz)

    def _sewer_load(self, far):
        cov, _, _ = self.envelope(far)
        c = cov * self.p.roof_c + (1 - cov) * self.c_perv
        storm = c * self.p.rainfall_in_per_hr * self.A / 43560.0
        sanitary = self.p.sanitary_peaking * self.p.sanitary_cfs_per_1000sqft * far * self.A / 1000.0
        return np.bincount(self.catch, weights=storm + sanitary, minlength=self.n_catch)

    def shaded(self, height: np.ndarray):
        """Boolean per lot: shaded at winter-solstice noon; and per-lot shadow count cast."""
        hi, hj = height[self.pair_i], height[self.pair_j]
        length = hi / self.tan_alt
        cast = (hi > hj) & (self.pair_dy <= length)
        shaded = np.zeros(self.n, dtype=bool)
        shaded[self.pair_j[cast]] = True
        return shaded, cast

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, far: np.ndarray) -> Dict:
        """Evaluate a FAR plan; returns per-lot arrays and scalar summaries."""
        p = self.p
        far = np.asarray(far, dtype=float)
        cov, floors, height = self.envelope(far)
        dfa = (far - self.far0) * self.A

        demand = self._demand(far)
        vc = demand / self.cap_taz
        tti = 1 + p.bpr_alpha * vc ** p.bpr_beta

        load = self._sewer_load(far)
        util = load / np.maximum(self.cap_catch, 1e-12)

        shaded, cast = self.shaded(height)
        base_shaded = self.baseline["shaded"] if hasattr(self, "baseline") else shaded
        new_shaded = shaded & ~base_shaded
        new_cast_pairs = cast & new_shaded[self.pair_j]
        shadow_imposed = np.bincount(self.pair_i[new_cast_pairs], minlength=self.n).astype(float)

        res_fa = far * self.A * self.res_share
        residents = res_fa / 1000.0 * p.persons_per_1000sqft
        res_cell = np.bincount(self.taz, weights=residents, minlength=self.n_taz)
        res_nbr = self._nbr @ res_cell
        green_pc_cell = self.green_area_nbr / np.maximum(res_nbr, 1.0)
        green_pc = green_pc_cell[self.taz]
        green_gini = gini(green_pc[self.is_res], residents[self.is_res])

        added = np.clip(dfa, 0, None)
        disp_share = float(added[self.vulnerable].sum() / added.sum()) if added.sum() > 0 else 0.0

        base = getattr(self, "baseline", None)
        taz_over = vc > p.vc_threshold
        vc0 = base["vc_taz"] if base else vc
        taz_violation = taz_over & (vc > vc0 + 1e-9)
        catch_over = util > 1.0 + 1e-9
        out = {
            # per-lot arrays
            "far": far, "coverage": cov, "height_ft": height, "delta_floor_area": dfa,
            "vc_lot": vc[self.taz], "tti_lot": tti[self.taz], "sewer_util_lot": util[self.catch],
            "shaded": shaded, "new_shaded": new_shaded, "shadow_imposed": shadow_imposed,
            "green_per_capita": green_pc, "residents": residents,
            # per-zone arrays
            "vc_taz": vc, "taz_over": taz_over, "taz_violation": taz_violation,
            "sewer_util": util, "catch_over": catch_over,
            # scalars
            "summary": {
                "added_floor_area_sqft": float(dfa.sum()),
                "added_residential_floor_area_sqft": float((dfa * self.res_share).sum()),
                "total_floor_area_sqft": float((far * self.A).sum()),
                "lots_changed": int((np.abs(far - self.far0) > 1e-9).sum()),
                "zoning_violations": int((far > self.ub + 1e-9).sum()),
                "lots_above_zoning_max_existing": int((self.far0 > self.max_far + 1e-9).sum()),
                "taz_over_capacity": int(taz_over.sum()),
                "taz_newly_over_capacity": int((taz_over & ~base["taz_over"]).sum()) if base else 0,
                "traffic_violations": int(taz_violation.sum()),
                "demand_weighted_tti": float((tti * demand).sum() / max(demand.sum(), 1e-9)),
                "max_vc": float(vc.max()),
                "catchments_over_capacity": int(catch_over.sum()),
                "max_sewer_utilisation": float(util.max()),
                "lots_newly_shaded": int(new_shaded.sum()),
                "solar_compliance": float(1.0 - new_shaded.sum() / self.n),
                "flood_zone_added_floor_area_sqft": float(added[self.flood].sum()),
                "green_space_gini": green_gini,
                "displacement_exposure_share": disp_share,
            },
        }
        return out

    def params_dict(self) -> Dict:
        return asdict(self.p)
