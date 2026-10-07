"""
Outcome model: social, economic and environmental consequences of a plan.

A plan adds floor area (sq ft) to each lot in four uses,
``plan[:, USE] = (residential, office, retail, community facility)``, within
the zoning envelope recorded in MapPLUTO (see :class:`ZoningEnvelope`). Every
outcome below is computed for the whole borough in one vectorised pass, so it
can serve as a learning signal at every step and as an objective for NSGA-III.

Social
    homes, affordable homes (City of Yes Universal Affordability Preference and
    Mandatory Inclusionary Housing), per-capita access to six everyday
    destination categories within a 15-minute walk (two-step floating
    catchment area, 2SFCA), share of residents with all six categories within
    15 minutes, displacement exposure (added floor area on vulnerable lots).
Economic
    jobs (LODES-calibrated densities), annual property tax (FY2026 class
    rates on assessed value), market value added, jobs-housing balance within
    walking distance, land-use mix within walking distance.
Environmental
    embodied carbon (CLF WBLCA v2 medians), operational carbon (LL84 2024
    medians of recent Manhattan buildings), 30-year life-cycle carbon, share of
    new floor area within a 15-minute walk of a subway station, floor area
    added in the 1% flood zone, plus the capacity screens (traffic, sewer,
    winter sunlight) of :class:`~pimaluos.physics.capacity.CapacityModel`.

Every parameter is either calibrated from data (``CityContext.params``) or
listed in :class:`OutcomeParams` with its source or marked as an assumption.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from pimaluos.context.build import CATEGORIES, CityContext
from pimaluos.physics.capacity import CapacityModel

USES = ("res", "office", "retail", "facility")
RES, OFFICE, RETAIL, FACILITY = range(4)
SQFT_PER_M2 = 10.7639
K_FOOD, K_TRANSIT = CATEGORIES.index("food"), CATEGORIES.index("transit")


@dataclass
class OutcomeParams:
    # Affordable housing.
    mih_affordable_share: float = 0.25      # MIH Option 1: 25% of residential floor area (ZR 23-154)
    # Property tax (NYC Department of Finance, Annual Report FY2026).
    assessment_ratio: float = 0.45          # class 2 and 4 assessed value / market value
    tax_rate_class2: float = 12.439         # per $100 AV, residential (class 2)
    tax_rate_class4: float = 10.848         # per $100 AV, commercial (class 4)
    affordable_tax_exempt: bool = True      # assumption: income-restricted floor area receives exemptions
    affordable_value_factor: float = 0.40   # assumption: value of income-restricted relative to market-rate floor area
    # Access. None = calibrate from existing conditions: food-store floor area /
    # retail floor area, and community-facility floor area / service sites.
    daily_needs_share_of_retail: Optional[float] = None
    facility_sqft_per_site: Optional[float] = None
    # Carbon.
    lifecycle_years: int = 30
    # Trip generation for added floor area (vehicle trips per 1,000 sq ft, peak hour).
    trip_rate_res: float = 0.15
    trip_rate_office: float = 0.50
    trip_rate_retail: float = 0.50
    trip_rate_facility: float = 0.30


class ZoningEnvelope:
    """Floor area each lot may add, by use (sq ft), from MapPLUTO 26v2.

    Residential: up to ResidFAR market-rate; between ResidFAR and AffResFAR only
    as income-restricted floor area (City of Yes Universal Affordability
    Preference). Commercial (office + retail): up to CommFAR. Community
    facility: up to FacilFAR. Total: up to the largest of the use maximums (or
    the existing FAR where higher). Existing floor area is never removed.
    """

    def __init__(self, gdf: pd.DataFrame, ctx: CityContext):
        g = gdf.reset_index(drop=True)
        A = g["lot_area_sqft"].to_numpy(float)
        f = lambda c: pd.to_numeric(g.get(c, 0), errors="coerce").fillna(0).to_numpy(float)  # noqa: E731
        resid, affres, comm, facil, manu = (f("max_resid_far"), f("max_affres_far"), f("max_comm_far"),
                                             f("max_facil_far"), f("max_manu_far"))
        affres = np.maximum(affres, resid)
        far0 = f("built_far")
        res0, fac0 = ctx.area0["res"], ctx.area0["facility"]
        com0 = f("com_area") - fac0
        self.A = A
        self.res_market = np.maximum(0.0, resid * A - res0)
        self.res_total = np.maximum(0.0, affres * A - res0)
        self.com = np.maximum(0.0, comm * A - np.maximum(com0, 0.0))
        self.facility = np.maximum(0.0, facil * A - fac0)
        ub_far = np.maximum.reduce([affres, comm, facil, manu, far0])
        self.total = np.maximum(0.0, (ub_far - far0) * A)
        self.ub_far = ub_far
        self.mih = g.get("mih_option", pd.Series(index=g.index, dtype=object)).notna().to_numpy() & (resid > 0)

    def use_caps(self) -> np.ndarray:
        return np.column_stack([self.res_total, self.com, self.com, self.facility])

    def clip(self, plan: np.ndarray) -> np.ndarray:
        """Project a plan into the envelope: per-use caps, a shared commercial cap
        for office + retail, and the total cap (scaled down proportionally)."""
        p = np.clip(np.asarray(plan, float), 0.0, None)
        p[:, RES] = np.minimum(p[:, RES], self.res_total)
        p[:, FACILITY] = np.minimum(p[:, FACILITY], self.facility)
        com = p[:, OFFICE] + p[:, RETAIL]
        s = np.where(com > self.com, self.com / np.maximum(com, 1e-12), 1.0)
        p[:, OFFICE] *= s
        p[:, RETAIL] *= s
        tot = p.sum(1)
        s = np.where(tot > self.total, self.total / np.maximum(tot, 1e-12), 1.0)
        return p * s[:, None]


class OutcomeModel:
    def __init__(self, gdf: pd.DataFrame, capacity: CapacityModel, ctx: CityContext,
                 params: Optional[OutcomeParams] = None):
        self.p = params or OutcomeParams()
        self.cap = capacity
        self.ctx = ctx
        self.env = ZoningEnvelope(gdf, ctx)
        # Plans may use the Universal Affordability Preference, so the capacity
        # model's zoning bound includes AffResFAR.
        capacity.ub = np.maximum(capacity.ub, self.env.ub_far)
        g = gdf.reset_index(drop=True)
        self.n = len(g)
        self.A = self.env.A
        self._set_params()
        self.node = ctx.node_of_lot
        self.n_nodes = ctx.n_nodes
        self.R = ctx.reach

        # Assessed value per sq ft by use, local median (traffic cell) with borough fallback.
        av = pd.to_numeric(g.get("assessed_total", 0), errors="coerce").fillna(0).to_numpy(float)
        bldg = pd.to_numeric(g.get("bldg_area_sqft", 0), errors="coerce").fillna(0).to_numpy(float)
        lu = g["land_use"].astype(str).str.zfill(2).to_numpy()
        avsf = np.where(bldg > 0, av / np.maximum(bldg, 1), np.nan)
        avsf = np.where((avsf > 1) & (avsf < 5000), avsf, np.nan)
        self.av_res = self._local_median(avsf, np.isin(lu, ["02", "03", "04"]))
        self.av_com = self._local_median(avsf, np.isin(lu, ["05"]))

        # Baseline per-capita access reference (resident-weighted median per category).
        self.base = None
        self.base = self.evaluate(np.zeros((self.n, 4)))

    def _set_params(self) -> None:
        """Derive the parameter-dependent arrays from ``self.ctx.params`` and ``self.p``."""
        cp, p, ctx = self.ctx.params, self.p, self.ctx
        self.unit_sqft = cp["unit_sqft_new"]
        self.ppu = cp["persons_per_unit"]
        self.epr = cp["employed_per_resident"]
        self.jobs_per_sqft = np.array([0.0, cp["jobs_per_ksf_office"], cp["jobs_per_ksf_retail"],
                                       cp["jobs_per_ksf_facility"]]) / 1000.0
        self.eci = np.array([cp[f"eci_{u}_kg_m2"] for u in USES]) / SQFT_PER_M2 / 1000.0  # t CO2e per sq ft
        self.opc = np.array([cp[f"opc_{u}_kg_ft2"] for u in USES]) / 1000.0              # t CO2e per sq ft per yr
        self.trip_rates = np.array([p.trip_rate_res, p.trip_rate_office, p.trip_rate_retail,
                                    p.trip_rate_facility]) / 1000.0
        k_svc = [CATEGORIES.index(c) for c in ("health", "civic")]
        k_ed = CATEGORIES.index("education")
        n_ed = ctx.meta.get("n_destinations", {}).get("education", int((ctx.supply0[:, k_ed] > 0).sum()))
        svc_sites = ctx.supply0[:, k_svc].sum() + n_ed
        if p.facility_sqft_per_site is None:
            p.facility_sqft_per_site = float(ctx.area0["facility"].sum() / max(svc_sites, 1.0))
        if p.daily_needs_share_of_retail is None:
            retail = ctx.area0["retail"].sum()
            share = ctx.supply0[:, K_FOOD].sum() / retail if retail > 0 else 0.5
            p.daily_needs_share_of_retail = float(min(1.0, share))

    def _init_like(self, other: "OutcomeModel", ctx: CityContext, params: OutcomeParams) -> "OutcomeModel":
        """Copy of ``other`` with different parameters (same lots, envelope and baseline)."""
        self.__dict__.update(other.__dict__)
        self.ctx, self.p = ctx, params
        self._set_params()
        return self

    def _local_median(self, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
        cell = self.cap.taz
        df = pd.DataFrame({"c": cell[mask], "v": v[mask]}).dropna()
        med = df.groupby("c")["v"].median()
        out = pd.Series(cell).map(med).to_numpy(float)
        glob = float(np.nanmedian(df["v"])) if len(df) else 1.0
        return np.where(np.isfinite(out), out, glob)

    # ------------------------------------------------------------------ helpers
    def _bc(self, w: np.ndarray) -> np.ndarray:
        return np.bincount(self.node, weights=w, minlength=self.n_nodes)

    def affordable(self, res_add: np.ndarray) -> np.ndarray:
        e = self.env
        uap = np.maximum(0.0, res_add - e.res_market)
        mkt_part = res_add - uap
        mih = np.where(e.mih, self.p.mih_affordable_share * mkt_part, 0.0)
        return uap + mih

    def far_of(self, plan: np.ndarray) -> np.ndarray:
        return self.cap.far0 + plan.sum(1) / np.maximum(self.A, 1e-9)

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, plan: np.ndarray) -> Dict:
        p, ctx = self.p, self.ctx
        plan = np.asarray(plan, float)
        add_tot = plan.sum(1)
        far = self.far_of(plan)
        cap = self.cap.evaluate(far, added_trips=plan @ self.trip_rates)

        res_add = plan[:, RES]
        aff = self.affordable(res_add)
        mkt = res_add - aff
        units = res_add / self.unit_sqft
        pop_new = units * self.ppu
        jobs_new = plan @ self.jobs_per_sqft

        # Node aggregates.
        P = self._bc(ctx.pop0 + pop_new)
        J = self._bc(ctx.jobs0 + jobs_new)
        W = self._bc(ctx.workers0 + pop_new * self.epr)
        S = ctx.supply0.copy()
        S[:, K_FOOD] += self._bc(plan[:, RETAIL] * p.daily_needs_share_of_retail)
        fac_sites = self._bc(plan[:, FACILITY] / p.facility_sqft_per_site)
        np.add.at(S, (np.arange(self.n_nodes), ctx.gap_service), fac_sites)

        # Two-step floating catchment area within the 15-minute walkshed.
        D = self.R @ P
        Racc = S / np.maximum(D, 1.0)[:, None]
        Acc = self.R @ Racc                                   # nodes x K, supply per resident
        if self.base is None:
            w = P / P.sum()
            ref = np.array([_weighted_median(Acc[:, k], w) for k in range(Acc.shape[1])])
            self.acc_ref = np.where(ref > 0, ref, np.maximum(Acc.mean(0), 1e-12))
        log_idx = np.log(np.maximum(Acc, 1e-6 * self.acc_ref) / self.acc_ref).mean(1)
        access_node = np.exp(log_idx)

        new_food = (self.R @ (self._bc(plan[:, RETAIL]) > 0).astype(float)) > 0
        has = ctx.has0.copy()
        has[:, K_FOOD] |= new_food
        fac_nodes = self._bc((plan[:, FACILITY] > 0).astype(float)) > 0
        for k in np.unique(ctx.gap_service):
            m = fac_nodes & (ctx.gap_service == k)
            has[:, k] |= (self.R @ m.astype(float)) > 0
        full15 = has.all(1)

        # Jobs-housing balance and land-use mix within the walkshed.
        Jw, Ww = self.R @ J, self.R @ W
        jh = np.where(np.maximum(Jw, Ww) > 0, np.minimum(Jw, Ww) / np.maximum(np.maximum(Jw, Ww), 1e-9), 0.0)
        areas = np.column_stack([self._bc(ctx.area0[u] + plan[:, i]) for i, u in enumerate(USES)])
        Fw = self.R @ areas
        sh = Fw / np.maximum(Fw.sum(1, keepdims=True), 1e-9)
        mix = -(sh * np.log(np.where(sh > 0, sh, 1.0))).sum(1) / np.log(len(USES))

        # Economics.
        tax = (mkt * self.av_res * p.tax_rate_class2 + (plan[:, OFFICE] + plan[:, RETAIL]) * self.av_com
               * p.tax_rate_class4 + (0.0 if p.affordable_tax_exempt else aff * self.av_res * p.affordable_value_factor
                                      * p.tax_rate_class2)) / 100.0
        value = (mkt * self.av_res + aff * self.av_res * p.affordable_value_factor
                 + (plan[:, OFFICE] + plan[:, RETAIL]) * self.av_com) / p.assessment_ratio

        # Carbon.
        emb = plan @ self.eci
        opc = plan @ self.opc
        transit_ok = ctx.has0[self.node, K_TRANSIT]

        Pn = P / max(P.sum(), 1e-9)
        lot_access = access_node[self.node]
        out = {
            "plan": plan, "capacity": cap, "far": far,
            # lot-level fields used by agents
            "lot": {"access": lot_access, "jh": jh[self.node], "mix": mix[self.node], "homes": units,
                    "affordable_homes": aff / self.unit_sqft, "jobs": jobs_new, "tax": tax, "value": value,
                    "embodied": emb, "operational": opc, "added": add_tot, "transit_ok": transit_ok,
                    "full15": full15[self.node]},
            "summary": {
                # social
                "homes_added": float(units.sum()),
                "affordable_homes_added": float(aff.sum() / self.unit_sqft),
                "affordable_share": float(aff.sum() / max(res_add.sum(), 1e-9)),
                "residents_added": float(pop_new.sum()),
                "access_index": float(np.exp((Pn * log_idx).sum())),
                "residents_full_15min_share": float((Pn * full15).sum()),
                "displacement_exposure_share": cap["summary"]["displacement_exposure_share"],
                "vulnerable_lot_added_floor_area_sqft": float(add_tot[self.cap.vulnerable].sum()),
                # economic
                "jobs_added": float(jobs_new.sum()),
                "tax_revenue_musd": float(tax.sum() / 1e6),
                "market_value_added_busd": float(value.sum() / 1e9),
                "jobs_housing_balance": float(((P + J) * jh).sum() / max((P + J).sum(), 1e-9)),
                "land_use_mix": float((Pn * mix).sum()),
                # environmental
                "embodied_carbon_kt": float(emb.sum() / 1e3),
                "operational_carbon_kt_per_yr": float(opc.sum() / 1e3),
                "lifecycle_carbon_kt": float((emb.sum() + p.lifecycle_years * opc.sum()) / 1e3),
                "transit_oriented_share": float(add_tot[transit_ok].sum() / max(add_tot.sum(), 1e-9)),
                "flood_zone_added_floor_area_sqft": cap["summary"]["flood_zone_added_floor_area_sqft"],
                "lots_newly_shaded": cap["summary"]["lots_newly_shaded"],
                "traffic_violations": cap["summary"]["traffic_violations"],
                "catchments_over_capacity": cap["summary"]["catchments_over_capacity"],
                # totals
                "added_floor_area_sqft": float(add_tot.sum()),
                **{f"added_{u}_sqft": float(plan[:, i].sum()) for i, u in enumerate(USES)},
            },
        }
        out["summary"]["capacity_violations"] = (out["summary"]["traffic_violations"]
                                                 + out["summary"]["catchments_over_capacity"]
                                                 + out["summary"]["lots_newly_shaded"])
        return out

    def params_dict(self) -> Dict:
        d = asdict(self.p)
        d.update({k: v for k, v in self.ctx.params.items()})
        return d


def _weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])
