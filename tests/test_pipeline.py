import json
from pathlib import Path

from pimaluos.experiments import load_config, run_all
from pimaluos.reporting import CONTEXT_MACROS, MACROS, make_report

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_smoke(tmp_path):
    cfg = load_config(str(ROOT / "configs" / "smoke.yaml"))
    cfg["synthetic"] = {"n_blocks_x": 3, "n_blocks_y": 3, "lots_per_block": 8, "seed": 0}
    cfg["seeds"] = [0]
    cfg["marl"]["iterations"] = 2
    out = run_all(cfg, tmp_path / "run")
    man = json.loads((out / "manifest.json").read_text())
    assert man["data"]["synthetic"] is True
    v = make_report(out, tmp_path / "gen")
    # Context macros need a real city; percentage differences are undefined when the comparator is zero.
    missing = [m for m in MACROS if m not in v and m not in CONTEXT_MACROS and m != "PlutoRelease"
               and not (m.startswith("PimVs") and m.endswith("Pct"))]
    assert missing == []
    for t in ["tab_social_economic_rows", "tab_environment_rows", "tab_uncertainty_rows", "tab_params_rows"]:
        assert (tmp_path / "gen" / f"{t}.tex").exists()
