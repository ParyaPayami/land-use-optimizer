import re
from pathlib import Path

from pimaluos.reporting import MACROS, write_pending

ROOT = Path(__file__).resolve().parents[1]


def test_pending_assets_match_registry(tmp_path):
    write_pending(tmp_path)
    for f in tmp_path.iterdir():
        committed = ROOT / "paper" / "pending" / f.name
        assert committed.exists(), f"missing paper/pending/{f.name}; run write_pending"
        assert committed.read_text() == f.read_text(), f"paper/pending/{f.name} is stale"


def test_manuscript_uses_only_registered_result_macros():
    tex = (ROOT / "paper" / "FINAL_SUBMISSION.tex").read_text()
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", tex))
    assert used <= set(MACROS), f"unregistered macros: {sorted(used - set(MACROS))}"
