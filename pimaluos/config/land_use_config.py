"""
Land Use Configuration Module

Defines land use categories, codes, and compatibility matrices for the PIMALUOS system.
"""

from typing import Dict, List

# Land Use Categories mapping
LAND_USE_CATEGORIES = {
    0: 'RESIDENTIAL',
    1: 'COMMERCIAL',
    2: 'INDUSTRIAL',
    3: 'MIXED_USE',
    4: 'PUBLIC',
    5: 'OPEN_SPACE'
}

LAND_USE_CODES = {v: k for k, v in LAND_USE_CATEGORIES.items()}

# Mapping for Dashboard Visualization (matches dashboard_app.py expectations)
DASHBOARD_LABEL_MAP = {
    'RESIDENTIAL': 'Residential',
    'COMMERCIAL': 'Commercial',
    'INDUSTRIAL': 'Industrial',
    'MIXED_USE': 'Mixed-Use',
    'PUBLIC': 'Public',
    'OPEN_SPACE': 'Open Space'
}

# Display colors for visualization
LAND_USE_COLORS = {
    0: '#F1C40F',  # Yellow - Residential
    1: '#E74C3C',  # Red - Commercial
    2: '#8E44AD',  # Purple - Industrial
    3: '#D35400',  # Orange - Mixed Use
    4: '#3498DB',  # Blue - Public/Institutional
    5: '#2ECC71'   # Green - Open Space
}

# Compatibility Matrix (0.0 to 1.0)
# Represents how compatible 'Row' use is when adjacent to 'Column' use.
# Used for utility calculation.
COMPATIBILITY_MATRIX = {
    #                 RES,  COM,  IND,  MIX,  PUB,  OPN
    'RESIDENTIAL':   [1.0, 0.6, 0.1, 0.8, 0.9, 1.0],
    'COMMERCIAL':    [0.7, 1.0, 0.6, 1.0, 0.8, 0.9],
    'INDUSTRIAL':    [0.1, 0.5, 1.0, 0.3, 0.4, 0.5],
    'MIXED_USE':     [0.9, 1.0, 0.4, 1.0, 0.9, 1.0],
    'PUBLIC':        [0.9, 0.8, 0.4, 0.9, 1.0, 1.0],
    'OPEN_SPACE':    [1.0, 1.0, 0.6, 1.0, 1.0, 1.0]
}

# Default target distribution for Manhattan-like city (can be overridden)
DEFAULT_TARGET_DISTRIBUTION = {
    'RESIDENTIAL': 0.40,
    'COMMERCIAL': 0.25,
    'INDUSTRIAL': 0.05,
    'MIXED_USE': 0.15,
    'PUBLIC': 0.10,
    'OPEN_SPACE': 0.05
}

def get_compatibility(use_a_idx: int, use_b_idx: int) -> float:
    """Get compatibility score between two land use codes."""
    cat_a = LAND_USE_CATEGORIES.get(use_a_idx, 'RESIDENTIAL')
    cat_b = LAND_USE_CATEGORIES.get(use_b_idx, 'RESIDENTIAL')
    
    # Check bounds
    if cat_a not in COMPATIBILITY_MATRIX:
        return 0.5
    
    # Get row
    row = COMPATIBILITY_MATRIX[cat_a]
    
    # Get column index - explicitly relying on ordered dict keys matches COMPATIBILITY_MATRIX structure 
    # But for safety let's map keys to indices
    keys = ['RESIDENTIAL', 'COMMERCIAL', 'INDUSTRIAL', 'MIXED_USE', 'PUBLIC', 'OPEN_SPACE']
    try:
        col_idx = keys.index(cat_b)
        return row[col_idx]
    except ValueError:
        return 0.5
