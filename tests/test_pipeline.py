import json
from pathlib import Path

from pimaluos.experiments import load_config, run_all
from pimaluos.reporting import MACROS, make_report

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
    missing = [m for m in MACROS if m not in v and not m.startswith("Rag") and m != "PlutoRelease"]
    assert missing == []
    assert (tmp_path / "gen" / "tab_main_rows.tex").exists()
