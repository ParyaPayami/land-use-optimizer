"""
Structured extraction of district-level bulk limits with RAG.

For each zoning district the extractor asks the LLM for the maximum residential,
commercial and community-facility FAR excluding bonuses (for height-factor
districts, the highest achievable value), matching the definition of MapPLUTO's
ResidFAR/CommFAR/FacilFAR, and validates the JSON answer with Pydantic. There
are no built-in fallback values: if extraction fails, the fields are ``None``
and the failure is recorded.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from pydantic import BaseModel

SYSTEM_PROMPT = (
    "You read excerpts of the New York City Zoning Resolution and extract numbers exactly as stated. "
    "Answer with a single JSON object and nothing else. Use null when the excerpts do not state a value."
)

QUERY_TEMPLATE = (
    "For zoning district {zone}, give the maximum floor area ratio (FAR) permitted for residential use, "
    "commercial use and community facility use, excluding any bonuses (plazas, arcades, inclusionary or "
    "other amenities). Where the residential FAR depends on height factor or open space ratio, give the "
    "highest achievable value. If a use is not permitted, use 0. "
    'Return JSON: {{"zone": "{zone}", "max_residential_far": number|null, '
    '"max_commercial_far": number|null, "max_community_facility_far": number|null, '
    '"section": string|null}}'
)

FIELDS = ["max_residential_far", "max_commercial_far", "max_community_facility_far"]


class DistrictLimits(BaseModel):
    zone: str
    max_residential_far: Optional[float] = None
    max_commercial_far: Optional[float] = None
    max_community_facility_far: Optional[float] = None
    section: Optional[str] = None
    parse_error: Optional[str] = None


def parse_json_object(text: str) -> Dict:
    """Extract the first balanced JSON object from model output."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found")


class ConstraintExtractor:
    def __init__(self, rag_pipeline, k: int = 6):
        if rag_pipeline is None:
            raise ValueError("ConstraintExtractor needs a RAGPipeline; there are no built-in defaults.")
        self.rag = rag_pipeline
        self.k = k

    def extract(self, zone: str) -> Dict:
        base = re.split(r"[/\s]", zone.strip())[0]
        out = self.rag.generate(QUERY_TEMPLATE.format(zone=base), system=SYSTEM_PROMPT, k=self.k, must_contain=base)
        try:
            data = parse_json_object(out["answer"])
            data["zone"] = zone
            limits = DistrictLimits(**{k: data.get(k) for k in ["zone", *FIELDS, "section"]})
        except Exception as e:  # noqa: BLE001 - record any parse/validation failure
            limits = DistrictLimits(zone=zone, parse_error=str(e)[:200])
        return {"limits": limits.model_dump(), "sources": out["sources"], "raw": out["answer"]}
