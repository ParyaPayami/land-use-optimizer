"""
NYC Zoning Compliance Module

Maps NYC zoning districts to allowed land uses based on NYC Zoning Resolution.
"""

from typing import Dict, List, Set
from pimaluos.config.land_use_config import LAND_USE_CATEGORIES

# NYC Zoning District to Allowed Land Uses Mapping
# Based on NYC Zoning Resolution (Updated December 2024 - includes "City of Yes" reforms)
# Source: NYC Department of City Planning - https://zola.planning.nyc.gov/
# Land use codes: 0=Residential, 1=Commercial, 2=Industrial, 3=Mixed-Use, 4=Public, 5=Open Space

ZONING_ALLOWED_USES = {
    # Residential Districts (Low to High Density)
    'R1': {0, 4, 5},  # Low-density detached: Residential, Public, Open Space only
    'R2': {0, 4, 5},  # Low-density attached
    'R3': {0, 3, 4, 5},  # Medium-density: + Mixed Use allowed
    'R4': {0, 3, 4, 5},  # Medium-density
    'R4A': {0, 3, 4, 5},  # New district from City of Yes (2024)
    'R4B': {0, 3, 4, 5},  # New district from City of Yes (2024)
    'R5': {0, 3, 4, 5},
    'R6': {0, 1, 3, 4, 5},  # Medium-high density: + Commercial (ground floor retail)
    'R7': {0, 1, 3, 4, 5},  # High-density
    'R8': {0, 1, 3, 4, 5},  # High-density
    'R9': {0, 1, 3, 4, 5},  # Very high-density
    'R10': {0, 1, 3, 4, 5},  # Very high-density
    'R11': {0, 1, 3, 4, 5},  # New district from City of Yes (2024)
    'R12': {0, 1, 3, 4, 5},  # New district from City of Yes (2024)
    
    # Commercial Districts
    'C1': {0, 1, 3, 4, 5},  # Local retail + residential above
    'C2': {0, 1, 3, 4, 5},  # Local service + residential
    'C3': {1, 3, 4, 5},  # Waterfront commercial (limited residential)
    'C4': {0, 1, 2, 3, 4, 5},  # Heavy commercial + light industrial
    'C5': {0, 1, 3, 4, 5},  # High-density commercial + residential
    'C6': {0, 1, 3, 4, 5},  # High-density commercial
    'C7': {1, 2, 3, 4, 5},  # Heavy commercial/industrial (limited residential)
    'C8': {1, 2, 3, 4, 5},  # General service (limited residential)
    
    # Manufacturing Districts
    'M1': {1, 2, 3, 4, 5},  # Light manufacturing + commercial (residential allowed in some M1 areas)
    'M2': {1, 2, 3, 4, 5},  # Medium manufacturing
    'M3': {2, 4, 5},  # Heavy manufacturing (NO residential - safety concerns)
    
    # Special Purpose Districts (more permissive for mixed-use development)
    'SP': {0, 1, 2, 3, 4, 5},  # Special districts - all uses typically allowed
    'MX': {0, 1, 3, 4, 5},  # Mixed-use districts
    'BPC': {0, 1, 3, 4, 5},  # Battery Park City
    'CD': {0, 1, 3, 4, 5},  # Coney Island
    'HS': {0, 1, 3, 4, 5},  # Hudson Square
    'LI': {1, 2, 3, 4, 5},  # Limited Industrial
}


def get_allowed_uses(zone_district: str) -> Set[int]:
    """
    Get allowed land use codes for a given zoning district.
    
    Args:
        zone_district: NYC zoning district code (e.g., 'R6', 'C1-5', 'M1-1')
        
    Returns:
        Set of allowed land use codes (0-5)
    """
    if not zone_district or zone_district == 'unknown':
        return {0, 1, 2, 3, 4, 5}  # Default: all uses allowed
    
    # Extract base zone (e.g., 'R6' from 'R6A', 'C1' from 'C1-5')
    base_zone = zone_district[:2].upper()
    
    # Handle special cases
    if base_zone.startswith('SP') or 'SPECIAL' in zone_district.upper():
        return ZONING_ALLOWED_USES['SPECIAL']
    
    # Look up in mapping
    if base_zone in ZONING_ALLOWED_USES:
        return ZONING_ALLOWED_USES[base_zone]
    
    # Default fallback for unknown zones
    return {0, 1, 3, 4, 5}  # Exclude heavy industrial by default


def is_land_use_allowed(land_use_code: int, zone_district: str) -> bool:
    """
    Check if a land use is allowed in a given zoning district.
    
    Args:
        land_use_code: Land use code (0-5)
        zone_district: NYC zoning district
        
    Returns:
        True if allowed, False otherwise
    """
    allowed = get_allowed_uses(zone_district)
    return land_use_code in allowed


def get_compliance_mask(land_uses: List[int], zone_districts: List[str]) -> List[bool]:
    """
    Get compliance mask for a list of land uses and zones.
    
    Args:
        land_uses: List of land use codes
        zone_districts: List of zoning districts (same length as land_uses)
        
    Returns:
        List of booleans indicating compliance
    """
    if len(land_uses) != len(zone_districts):
        raise ValueError("land_uses and zone_districts must have same length")
    
    return [
        is_land_use_allowed(use, zone)
        for use, zone in zip(land_uses, zone_districts)
    ]


def count_violations(land_uses: List[int], zone_districts: List[str]) -> int:
    """
    Count zoning violations.
    
    Args:
        land_uses: List of land use codes
        zone_districts: List of zoning districts
        
    Returns:
        Number of violations
    """
    compliance = get_compliance_mask(land_uses, zone_districts)
    return sum(1 for c in compliance if not c)
