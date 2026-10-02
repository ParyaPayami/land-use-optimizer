"""
RAG extraction benchmark against MapPLUTO.

Reference values: for every zoning district (``ZoneDist1``) in the study area,
the modal ``ResidFAR``, ``CommFAR`` and ``FacilFAR`` published by NYC DCP in
MapPLUTO, computed over lots that are not in a special purpose district, have
no commercial overlay and are not split between districts. Districts with fewer
than ``min_lots`` such lots are skipped. The share of lots that agree with the
mode is reported as a check on the reference itself.

Prediction: :class:`pimaluos.knowledge.ConstraintExtractor` over the Zoning
Resolution text supplied in ``--zr-dir`` (the version must match the MapPLUTO
release; record both).

Metrics per field: coverage (non-null answers), exact accuracy (|error| <=
0.01), mean absolute error over answered districts, with Wilson 95 % intervals
for accuracy. Every prompt, answer and retrieved source is saved.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pimaluos.core.data_loader import get_data_loader
from pimaluos.knowledge import FIELDS, ConstraintExtractor, DocumentLoader, RAGPipeline, get_llm

PLUTO_FIELDS = {"max_residential_far": "max_resid_far", "max_commercial_far": "max_comm_far",
                "max_community_facility_far": "max_facil_far"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def pluto_reference(gdf: pd.DataFrame, min_lots: int = 5) -> pd.DataFrame:
    g = gdf.copy()
    clean = pd.Series(True, index=g.index)
    for col in ["special_district", "overlay"]:
        if col in g:
            clean &= g[col].isna() | (g[col].astype(str).str.strip().isin(["", "None", "nan"]))
    if "split_zone" in g:
        clean &= g["split_zone"].astype(str).str.upper().ne("Y")
    g = g[clean & g["zone_district"].ne("UNKNOWN")]
    rows = []
    for zone, grp in g.groupby("zone_district"):
        if len(grp) < min_lots:
            continue
        row = {"zone": zone, "n_lots": int(len(grp))}
        for f, col in PLUTO_FIELDS.items():
            vals = grp[col].round(2)
            mode = float(vals.mode().iloc[0])
            row[f] = mode
            row[f"{f}_agreement"] = float((vals == mode).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def score(ref: pd.DataFrame, pred: pd.DataFrame, tol: float = 0.01) -> dict:
    m = ref.merge(pred, on="zone", suffixes=("_ref", "_pred"))
    out = {"n_districts": int(len(m))}
    for f in FIELDS:
        r, p = m[f"{f}_ref"].astype(float), pd.to_numeric(m[f"{f}_pred"], errors="coerce")
        answered = p.notna()
        correct = (np.abs(p - r) <= tol) & answered
        k, n = int(correct.sum()), int(len(m))
        lo, hi = wilson(k, n)
        out[f] = {"coverage": float(answered.mean()) if n else math.nan, "accuracy": k / n if n else math.nan,
                  "accuracy_ci95": [lo, hi],
                  "mae_answered": float(np.abs(p - r)[answered].mean()) if answered.any() else math.nan}
    allk = sum(int(((np.abs(pd.to_numeric(m[f"{f}_pred"], errors="coerce") - m[f"{f}_ref"]) <= tol)).sum())
               for f in FIELDS)
    alln = 3 * len(m)
    out["overall_accuracy"] = allk / alln if alln else math.nan
    out["overall_accuracy_ci95"] = list(wilson(allk, alln))
    return out


def run_rag_benchmark(pluto_path: str, zr_dir: str, provider: str, model: Optional[str], out: Path,
                      city: str = "manhattan") -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = get_data_loader(city, pluto_path).load()
    ref = pluto_reference(ds.gdf)
    ref.to_csv(out / "reference.csv", index=False)
    llm = get_llm(provider, model)
    rag = RAGPipeline(llm, cache_path=out / "llm_cache.json")
    n_chunks = rag.index(DocumentLoader.load_directory(Path(zr_dir)))
    ex = ConstraintExtractor(rag)
    preds, log = [], []
    for zone in ref["zone"]:
        r = ex.extract(zone)
        preds.append(r["limits"])
        log.append({"zone": zone, **r})
    pred = pd.DataFrame(preds)
    pred.to_csv(out / "predictions.csv", index=False)
    (out / "extraction_log.json").write_text(json.dumps(log, indent=1, default=str))
    res = score(ref, pred)
    res.update({"provider": llm.name, "n_chunks": n_chunks, "vector_backend": rag.store.backend,
                "reference_mean_agreement": {f: float(ref[f"{f}_agreement"].mean()) for f in FIELDS},
                "parse_failures": int(pred["parse_error"].notna().sum())})
    (out / "rag_results.json").write_text(json.dumps(res, indent=1))
    return res
