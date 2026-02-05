"""
PIMALUOS Zoning Constraint Parser

Extracts structured constraints from zoning regulations:
- Pydantic models for type-safe constraint output
- Multi-city support with city-specific parsers
- Caching layer for extracted constraints
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import re

from pydantic import BaseModel, Field


# ===== Pydantic Models =====

class UseRegulations(BaseModel):
    """Permitted and prohibited uses."""
    permitted_as_of_right: List[str] = Field(default_factory=list)
    permitted_with_special_permit: List[str] = Field(default_factory=list)
    prohibited: List[str] = Field(default_factory=list)


class BulkRegulations(BaseModel):
    """Bulk and density regulations."""
    max_far: Optional[float] = None
    max_far_with_bonus: Optional[float] = None
    community_facility_far: Optional[float] = None
    max_height_ft: Optional[float] = None
    max_stories: Optional[int] = None
    max_lot_coverage: Optional[float] = None


class LotRequirements(BaseModel):
    """Minimum lot requirements."""
    min_lot_area_sqft: Optional[float] = None
    min_lot_width_ft: Optional[float] = None
    min_lot_depth_ft: Optional[float] = None


class YardRequirements(BaseModel):
    """Yard/setback requirements."""
    front_yard_ft: Optional[float] = None
    side_yard_ft: Optional[float] = None
    rear_yard_ft: Optional[float] = None
    min_distance_between_buildings_ft: Optional[float] = None


class ParkingRequirements(BaseModel):
    """Parking requirements."""
    residential_spaces_per_unit: Optional[float] = None
    commercial_spaces_per_sqft: Optional[float] = None
    retail_spaces_per_sqft: Optional[float] = None
    max_parking_reduction_transit: Optional[float] = None


class InclusionaryHousing(BaseModel):
    """Inclusionary housing requirements."""
    required: bool = False
    affordable_percentage: Optional[float] = None
    ami_threshold: Optional[float] = None
    bonus_far: Optional[float] = None


class ZoningConstraints(BaseModel):
    """Complete zoning constraints for a district."""
    zone_code: str
    zone_type: str = "residential"  # residential, commercial, manufacturing, mixed
    display_name: Optional[str] = None
    
    uses: UseRegulations = Field(default_factory=UseRegulations)
    bulk: BulkRegulations = Field(default_factory=BulkRegulations)
    lot: LotRequirements = Field(default_factory=LotRequirements)
    yards: YardRequirements = Field(default_factory=YardRequirements)
    parking: ParkingRequirements = Field(default_factory=ParkingRequirements)
    inclusionary: InclusionaryHousing = Field(default_factory=InclusionaryHousing)
    
    special_conditions: List[str] = Field(default_factory=list)
    source_documents: List[str] = Field(default_factory=list)
    extracted_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


# ===== Constraint Cache =====

class ConstraintCache:
    """Persistent cache for extracted constraints."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("./data/constraint_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, ZoningConstraints] = {}
        self._load()
    
    def _cache_file(self) -> Path:
        return self.cache_dir / "constraints.json"
    
    def _load(self) -> None:
        if self._cache_file().exists():
            with open(self._cache_file()) as f:
                data = json.load(f)
            self.cache = {k: ZoningConstraints(**v) for k, v in data.items()}
    
    def _save(self) -> None:
        data = {k: v.dict() for k, v in self.cache.items()}
        with open(self._cache_file(), "w") as f:
            json.dump(data, f, default=str, indent=2)
    
    def get(self, zone_code: str) -> Optional[ZoningConstraints]:
        return self.cache.get(zone_code)
    
    def set(self, zone_code: str, constraints: ZoningConstraints) -> None:
        self.cache[zone_code] = constraints
        self._save()
    
    def has(self, zone_code: str) -> bool:
        return zone_code in self.cache
    
    def all(self) -> Dict[str, ZoningConstraints]:
        return self.cache.copy()


# ===== Constraint Extractor =====

class ConstraintExtractor:
    """
    Extract structured constraints from zoning regulations.
    
    Uses LLM with RAG to parse zoning documents and extract
    computable constraints in Pydantic models.
    """
    
    # Default constraints by zone prefix
    DEFAULTS = {
        "R6": {"max_far": 2.0, "max_height_ft": 65, "max_stories": 6},
        "R7": {"max_far": 3.44, "max_height_ft": 85, "max_stories": 8},
        "R8": {"max_far": 6.02, "max_height_ft": 120, "max_stories": 12},
        "R9": {"max_far": 7.52, "max_height_ft": 145, "max_stories": 15},
        "R10": {"max_far": 10.0, "max_height_ft": 210, "max_stories": 21},
        "C1": {"max_far": 1.0, "max_height_ft": 65},
        "C2": {"max_far": 2.0, "max_height_ft": 85},
        "C4": {"max_far": 3.4, "max_height_ft": 125},
        "C5": {"max_far": 10.0, "max_height_ft": None},
        "C6": {"max_far": 10.0, "max_height_ft": None},
        "M1": {"max_far": 2.0, "max_height_ft": 60},
        "M2": {"max_far": 2.0, "max_height_ft": 60},
        "M3": {"max_far": 2.0, "max_height_ft": None},
    }
    
    def __init__(
        self, 
        rag_pipeline = None,
        cache: ConstraintCache = None
    ):
        self.rag = rag_pipeline
        self.cache = cache or ConstraintCache()
    
    def extract_for_zone(
        self, 
        zone_code: str,
        force_refresh: bool = False
    ) -> ZoningConstraints:
        """
        Extract constraints for a zoning district.
        
        Args:
            zone_code: NYC zoning code (e.g., 'R6', 'C4-5')
            force_refresh: Bypass cache
            
        Returns:
            ZoningConstraints model
        """
        # Check cache
        if not force_refresh and self.cache.has(zone_code):
            return self.cache.get(zone_code)
        
        print(f"Extracting constraints for {zone_code}...")
        
        # Use RAG if available
        if self.rag:
            constraints = self._extract_via_llm(zone_code)
        else:
            constraints = self._get_defaults(zone_code)
        
        # Cache result
        self.cache.set(zone_code, constraints)
        
        return constraints
    
    def _extract_via_llm(self, zone_code: str) -> ZoningConstraints:
        """Extract constraints using LLM with RAG."""
        query = f"""Extract detailed zoning regulations for {zone_code} district.

Provide the following in JSON format:
- permitted_uses: list of permitted uses
- prohibited_uses: list of prohibited uses
- max_far: maximum floor area ratio (number)
- max_height_ft: maximum building height in feet (number)
- max_stories: maximum stories (integer)
- min_lot_area_sqft: minimum lot area (number)
- front_yard_ft, side_yard_ft, rear_yard_ft: yard requirements (numbers)
- parking_ratio: parking spaces per dwelling unit (number)

Return ONLY valid JSON."""
        
        response = self.rag.generate(query)
        
        # Parse JSON from response
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return self._dict_to_constraints(zone_code, data)
        except json.JSONDecodeError:
            pass
        
        # Fallback to defaults
        return self._get_defaults(zone_code)
    
    def _dict_to_constraints(self, zone_code: str, data: Dict) -> ZoningConstraints:
        """Convert extracted dictionary to ZoningConstraints."""
        zone_type = self._infer_zone_type(zone_code)
        
        return ZoningConstraints(
            zone_code=zone_code,
            zone_type=zone_type,
            uses=UseRegulations(
                permitted_as_of_right=data.get("permitted_uses", []),
                prohibited=data.get("prohibited_uses", []),
            ),
            bulk=BulkRegulations(
                max_far=data.get("max_far"),
                max_height_ft=data.get("max_height_ft"),
                max_stories=data.get("max_stories"),
            ),
            lot=LotRequirements(
                min_lot_area_sqft=data.get("min_lot_area_sqft"),
                min_lot_width_ft=data.get("min_lot_width_ft"),
            ),
            yards=YardRequirements(
                front_yard_ft=data.get("front_yard_ft"),
                side_yard_ft=data.get("side_yard_ft"),
                rear_yard_ft=data.get("rear_yard_ft"),
            ),
            parking=ParkingRequirements(
                residential_spaces_per_unit=data.get("parking_ratio"),
            ),
            extracted_at=datetime.now(),
        )
    
    def _get_defaults(self, zone_code: str) -> ZoningConstraints:
        """Get default constraints based on zone prefix."""
        # Find matching prefix
        prefix = None
        for p in sorted(self.DEFAULTS.keys(), key=len, reverse=True):
            if zone_code.startswith(p):
                prefix = p
                break
        
        defaults = self.DEFAULTS.get(prefix, {"max_far": 2.0, "max_height_ft": 65})
        zone_type = self._infer_zone_type(zone_code)
        
        return ZoningConstraints(
            zone_code=zone_code,
            zone_type=zone_type,
            uses=UseRegulations(
                permitted_as_of_right=["residential"] if zone_type == "residential" else ["commercial"],
            ),
            bulk=BulkRegulations(
                max_far=defaults.get("max_far"),
                max_height_ft=defaults.get("max_height_ft"),
                max_stories=defaults.get("max_stories"),
            ),
            lot=LotRequirements(
                min_lot_area_sqft=1700,
                min_lot_width_ft=18,
            ),
            yards=YardRequirements(
                front_yard_ft=10,
                side_yard_ft=8,
                rear_yard_ft=30,
            ),
            parking=ParkingRequirements(
                residential_spaces_per_unit=0.5,
            ),
            extracted_at=datetime.now(),
        )
    
    def _infer_zone_type(self, zone_code: str) -> str:
        """Infer zone type from code."""
        if zone_code.startswith("R"):
            return "residential"
        elif zone_code.startswith("C"):
            return "commercial"
        elif zone_code.startswith("M"):
            return "manufacturing"
        return "mixed"
    
    def validate_proposal(
        self, 
        zone_code: str,
        proposal: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a development proposal against zoning constraints.
        
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        constraints = self.extract_for_zone(zone_code)
        violations = []
        
        # Check FAR
        if proposal.get("far") and constraints.bulk.max_far:
            if proposal["far"] > constraints.bulk.max_far:
                violations.append(
                    f"FAR {proposal['far']:.2f} exceeds max {constraints.bulk.max_far:.2f}"
                )
        
        # Check height
        if proposal.get("height_ft") and constraints.bulk.max_height_ft:
            if proposal["height_ft"] > constraints.bulk.max_height_ft:
                violations.append(
                    f"Height {proposal['height_ft']}ft exceeds max {constraints.bulk.max_height_ft}ft"
                )
        
        # Check stories
        if proposal.get("stories") and constraints.bulk.max_stories:
            if proposal["stories"] > constraints.bulk.max_stories:
                violations.append(
                    f"Stories {proposal['stories']} exceeds max {constraints.bulk.max_stories}"
                )
        
        # Check use
        proposed_use = proposal.get("use", "residential")
        if constraints.uses.prohibited and proposed_use in constraints.uses.prohibited:
            violations.append(f"Use '{proposed_use}' is prohibited")
        
        return len(violations) == 0, violations


# ===== City-Specific Parsers =====

class NYCZoningParser(ConstraintExtractor):
    """NYC-specific zoning parser with special district handling."""
    
    SPECIAL_DISTRICTS = {
        "HY": "Hudson Yards",
        "TB": "TriBeCa",
        "LM": "Lower Manhattan",
        "SG": "Special Garment Center",
    }
    
    def _get_defaults(self, zone_code: str) -> ZoningConstraints:
        constraints = super()._get_defaults(zone_code)
        
        # Handle special districts
        for prefix, name in self.SPECIAL_DISTRICTS.items():
            if prefix in zone_code:
                constraints.display_name = name
                constraints.special_conditions.append(f"Special {name} District regulations apply")
        
        return constraints


class ChicagoZoningParser(ConstraintExtractor):
    """Chicago-specific zoning parser."""
    
    DEFAULTS = {
        "RS-3": {"max_far": 0.9, "max_height_ft": 30},
        "RT-4": {"max_far": 1.2, "max_height_ft": 38},
        "RM-5": {"max_far": 2.0, "max_height_ft": 45},
        "B1-1": {"max_far": 1.2, "max_height_ft": 32},
        "C1-2": {"max_far": 2.2, "max_height_ft": 50},
        "DX-7": {"max_far": 5.0, "max_height_ft": 70},
    }


# Example usage
if __name__ == "__main__":
    # Test without RAG
    extractor = ConstraintExtractor()
    
    # Extract for test zones
    for zone in ["R6", "R7", "C4-5", "M1-4"]:
        constraints = extractor.extract_for_zone(zone)
        print(f"\n{zone}:")
        print(f"  Max FAR: {constraints.bulk.max_far}")
        print(f"  Max Height: {constraints.bulk.max_height_ft}")
        print(f"  Type: {constraints.zone_type}")
    
    # Test validation
    proposal = {"far": 3.0, "height_ft": 80, "use": "residential"}
    is_valid, violations = extractor.validate_proposal("R6", proposal)
    print(f"\nProposal valid for R6: {is_valid}")
    if violations:
        print(f"  Violations: {violations}")
