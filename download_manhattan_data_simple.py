#!/usr/bin/env python3
"""
Simple Manhattan data downloader (minimal dependencies).

Downloads NYC MapPLUTO data without requiring full PIMALUOS installation.
"""

import requests
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NYC Department of City Planning Direct Download URLs
PLUTO_URL = "https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/mappluto/nyc_mappluto_25v3_1_shp.zip"
ZONING_URL = "https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/gis-zoning-features/nycgiszoningfeatures_202511shp.zip"

def download_manhattan_data():
    """Download Manhattan MapPLUTO data."""
    
    logger.info("=" * 70)
    logger.info("  MANHATTAN DATA DOWNLOAD (Simplified)")
    logger.info("  NYC MapPLUTO Official Dataset")
    logger.info("=" * 70)
    
    # Create data directory
    data_dir = Path("./data/manhattan")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download MapPLUTO
        logger.info("\nDownloading MapPLUTO data from NYC Open Data...")
        logger.info("This may take several minutes...")
        
        response = requests.get(PLUTO_URL, stream=True, timeout=300)
        response.raise_for_status()
        
        logger.info("Download complete. Extracting files...")
        
        with ZipFile(BytesIO(response.content)) as z:
            z.extractall(data_dir / 'pluto_raw')
        
        logger.info(f"✓ MapPLUTO data extracted to {data_dir / 'pluto_raw'}")
        
        # Download Zoning
        logger.info("\nDownloading zoning data...")
        response = requests.get(ZONING_URL, stream=True, timeout=300)
        response.raise_for_status()
        
        with ZipFile(BytesIO(response.content)) as z:
            z.extractall(data_dir / 'zoning_raw')
        
        logger.info(f"✓ Zoning data extracted to {data_dir / 'zoning_raw'}")
        
        # List downloaded files
        pluto_files = list((data_dir / 'pluto_raw').glob('*.shp'))
        zoning_files = list((data_dir / 'zoning_raw').glob('*.shp'))
        
        logger.info("\n" + "=" * 70)
        logger.info("  DOWNLOAD COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"\nDownloaded files:")
        logger.info(f"  MapPLUTO: {len(pluto_files)} shapefiles")
        logger.info(f"  Zoning: {len(zoning_files)} shapefiles")
        logger.info(f"\nData directory: {data_dir.absolute()}")
        
        logger.info(f"\n✓ Manhattan data ready!")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Install dependencies: pip install -r requirements.txt")
        logger.info(f"  2. Run demo: python demo_complete_pipeline.py")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        logger.error(f"\nFor manual download:")
        logger.error(f"  MapPLUTO: {PLUTO_URL}")
        logger.error(f"  Zoning: {ZONING_URL}")
        return False

if __name__ == "__main__":
    import sys
    success = download_manhattan_data()
    sys.exit(0 if success else 1)
