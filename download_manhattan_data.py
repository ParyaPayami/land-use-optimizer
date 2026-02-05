#!/usr/bin/env python3
"""
Download Manhattan MapPLUTO data from NYC Open Data.

This script downloads the official NYC MapPLUTO dataset for Manhattan
and prepares it for use with PIMALUOS.
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_manhattan_data():
    """Download Manhattan parcel data."""
    
    logger.info("=" * 70)
    logger.info("  MANHATTAN DATA DOWNLOAD")
    logger.info("  NYC MapPLUTO Official Dataset")
    logger.info("=" * 70)
    
    try:
        from pimaluos.core import get_data_loader
        
        # Create data loader
        logger.info("\nInitializing Manhattan data loader...")
        loader = get_data_loader('manhattan', cache=True)
        
        # Download and process data
        logger.info("\nDownloading data from NYC Open Data...")
        logger.info("This may take several minutes depending on your connection...")
        
        gdf, features = loader.load_and_compute_features()
        
        logger.info("\n" + "=" * 70)
        logger.info("  DOWNLOAD COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"\nData Summary:")
        logger.info(f"  Total parcels: {len(gdf):,}")
        logger.info(f"  Features computed: {len(features.columns)}")
        logger.info(f"  Data directory: {loader.data_dir}")
        
        # Show sample
        logger.info(f"\nSample parcels:")
        if 'address' in gdf.columns and 'zone_district' in gdf.columns:
            sample_cols = ['address', 'zone_district', 'land_use', 'bldg_area_sqft', 'num_floors']
            available_cols = [col for col in sample_cols if col in gdf.columns]
            logger.info(f"\n{gdf[available_cols].head(5).to_string()}")
        
        logger.info(f"\nFeature statistics:")
        logger.info(f"\n{features.describe().to_string()}")
        
        logger.info(f"\n✓ Manhattan data ready for use with PIMALUOS!")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Run demo: python demo_complete_pipeline.py")
        logger.info(f"  2. Or use in your own script:")
        logger.info(f"     from pimaluos import UrbanOptSystem")
        logger.info(f"     system = UrbanOptSystem(city='manhattan')")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Error downloading data: {e}")
        logger.error(f"\nTroubleshooting:")
        logger.error(f"  1. Check internet connection")
        logger.error(f"  2. Verify NYC Open Data is accessible")
        logger.error(f"  3. Try again later if server is busy")
        logger.error(f"\nFor manual download:")
        logger.error(f"  URL: https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks")
        
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = download_manhattan_data()
    sys.exit(0 if success else 1)
