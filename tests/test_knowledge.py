import pandas as pd
import pytest

from pimaluos.knowledge import ConstraintExtractor, Document, MockLLM, RAGPipeline, TextSplitter, parse_json_object
from pimaluos.rag_benchmark import pluto_reference, score, wilson


def test_parse_json_object_handles_prose_and_nesting():
    assert parse_json_object('Answer: {"a": {"b": 1}, "c": 2} done') == {"a": {"b": 1}, "c": 2}
    with pytest.raises(ValueError):
        parse_json_object("no json")


def test_splitter_overlap():
    chunks = TextSplitter(100, 20).split(Document("x" * 450))
    assert len(chunks) >= 5 and all(len(c.content) <= 100 for c in chunks)


def test_extractor_requires_rag_and_parses():
    with pytest.raises(ValueError):
        ConstraintExtractor(None)
    llm = MockLLM(response={"max_residential_far": 6.02, "max_commercial_far": 0, "max_community_facility_far": 6.5})
    rag = RAGPipeline(llm)
    rag.index([Document("R8 districts: the maximum residential floor area ratio is 6.02."), Document("C6-4 ...")])
    r = ConstraintExtractor(rag).extract("R8")
    assert r["limits"]["max_residential_far"] == 6.02 and r["limits"]["parse_error"] is None


def test_reference_and_scoring():
    gdf = pd.DataFrame({"zone_district": ["R8"] * 6 + ["C6-4"] * 6,
                        "max_resid_far": [6.02] * 5 + [7.2] + [10.0] * 6,
                        "max_comm_far": [0.0] * 6 + [10.0] * 6, "max_facil_far": [6.5] * 6 + [10.0] * 6,
                        "special_district": [None] * 12, "overlay": [None] * 12, "split_zone": ["N"] * 12})
    ref = pluto_reference(gdf)
    assert set(ref.zone) == {"R8", "C6-4"}
    assert ref.set_index("zone").loc["R8", "max_residential_far"] == 6.02
    pred = pd.DataFrame({"zone": ["R8", "C6-4"], "max_residential_far": [6.02, None],
                         "max_commercial_far": [0, 10], "max_community_facility_far": [6.5, 9.0]})
    s = score(ref, pred)
    assert s["max_residential_far"]["accuracy"] == 0.5 and s["max_residential_far"]["coverage"] == 0.5
    assert abs(s["overall_accuracy"] - 4 / 6) < 1e-9
    lo, hi = wilson(4, 6)
    assert lo < 4 / 6 < hi
