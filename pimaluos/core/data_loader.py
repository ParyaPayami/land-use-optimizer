"""
PIMALUOS Data Loader Module

Provides abstract base class and city-specific implementations for loading
parcel data from various open data sources.

Supported cities:
- Manhattan, NYC (MapPLUTO)
- Chicago, IL (Cook County Assessor)
- Los Angeles, CA (LA County)
- Boston, MA (Boston GIS)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from sklearn.preprocessing import StandardScaler

from pimaluos.config.settings import get_city_config, CityConfig

warnings.filterwarnings('ignore')


class CityDataLoader(ABC):
    """
    Abstract base class for city-specific parcel data loaders.
    
    Subclasses must implement:
        - download_parcels(): Download raw parcel data
        - download_zoning(): Download zoning district data
        - get_column_mapping(): Map source columns to standard schema
    """
    
    # Standard column schema that all city loaders must map to
    STANDARD_COLUMNS = {
        'parcel_id': 'Unique parcel identifier',
        'address': 'Street address',
        'lot_area_sqft': 'Lot area in square feet',
        'bldg_area_sqft': 'Building area in square feet',
        'num_floors': 'Number of floors',
        'year_built': 'Year built',
        'land_use': 'Land use category code',
        'zone_district': 'Zoning district',
        'built_far': 'Current floor area ratio',
        'max_far': 'Maximum allowed FAR',
        'assessed_total': 'Total assessed value',
        'assessed_land': 'Land assessed value',
        'units_residential': 'Number of residential units',
        'units_total': 'Total units',
    }
    
    def __init__(self, city: str, data_dir: Optional[Path] = None, cache: bool = True):
        """
        Initialize data loader.
        
        Args:
            city: City identifier (e.g., 'manhattan', 'chicago')
            data_dir: Base data directory (default: ./data/{city})
            cache: Whether to cache downloaded data
        """
        self.city = city
        self.config = get_city_config(city)
        self.data_dir = Path(data_dir) if data_dir else Path(f'./data/{city}')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        
        # Data containers
        self.parcels_gdf: Optional[gpd.GeoDataFrame] = None
        self.zoning_gdf: Optional[gpd.GeoDataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.features_normalized: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def download_parcels(self) -> gpd.GeoDataFrame:
        """Download and return parcel data."""
        pass
    
    @abstractmethod
    def download_zoning(self) -> gpd.GeoDataFrame:
        """Download and return zoning district data."""
        pass
    
    @abstractmethod
    def get_column_mapping(self) -> Dict[str, str]:
        """
        Return mapping from source columns to standard schema.
        
        Returns:
            Dict mapping source column names to STANDARD_COLUMNS keys
        """
        pass
    
    def standardize_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Apply column mapping to standardize schema."""
        mapping = self.get_column_mapping()
        return gdf.rename(columns=mapping)
    
    def compute_node_features(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Compute 47-dimensional feature vector for each parcel.
        
        This method implements the standard feature engineering pipeline
        that can be used across all cities.
        
        Args:
            gdf: Standardized parcel GeoDataFrame
            
        Returns:
            DataFrame with computed features
        """
        print("Computing node features...")
        features = pd.DataFrame(index=gdf.index)
        
        # ===== PHYSICAL ATTRIBUTES (8 features) =====
        features['area_sqft'] = gdf.geometry.area
        features['perimeter'] = gdf.geometry.length
        features['shape_complexity'] = (
            (gdf.geometry.length ** 2) / (4 * np.pi * gdf.geometry.area + 1e-10)
        )
        
        features['built_floor_area'] = gdf['bldg_area_sqft'].fillna(0)
        features['num_floors'] = gdf['num_floors'].fillna(0)
        features['year_built'] = gdf['year_built'].fillna(gdf['year_built'].median())
        features['year_altered'] = gdf.get('year_altered', gdf['year_built'])
        features['bldg_class_encoded'] = pd.Categorical(
            gdf.get('bldg_class', 'UNKNOWN')
        ).codes
        
        # ===== CURRENT LAND USE (6 features) =====
        features['current_far'] = gdf['built_far'].fillna(0)
        features['max_far'] = gdf['max_far'].fillna(0)
        features['lot_coverage'] = (
            gdf['bldg_area_sqft'].fillna(0) / gdf['lot_area_sqft'].replace(0, 1)
        ).clip(0, 1)
        
        # Land use category encoding
        features['landuse_encoded'] = pd.Categorical(gdf['land_use']).codes
        features['zone_encoded'] = pd.Categorical(gdf['zone_district']).codes
        
        # Land use one-hot encoding
        landuse_dummies = pd.get_dummies(gdf['land_use'], prefix='lu')
        
        # ===== ACCESSIBILITY METRICS (8 features) =====
        centroid = gdf.geometry.centroid
        city_center_lat = self.config.latitude
        city_center_lon = self.config.longitude
        
        features['dist_to_center'] = np.sqrt(
            (centroid.y - city_center_lat)**2 + 
            (centroid.x - city_center_lon)**2
        ) * 111000  # Convert degrees to meters
        
        # Placeholders for network metrics (computed after graph building)
        features['dist_to_subway'] = 0
        features['dist_to_bus'] = 0
        features['dist_to_park'] = 0
        features['betweenness_centrality'] = 0
        features['closeness_centrality'] = 0
        features['degree_centrality'] = 0
        features['clustering_coefficient'] = 0
        
        # ===== ENVIRONMENTAL INDICATORS (5 features) =====
        features['tree_canopy_pct'] = 0  # Placeholder
        features['impervious_pct'] = (features['lot_coverage'] * 100).clip(0, 100)
        features['flood_zone'] = gdf.get('flood_zone', 0).fillna(0).astype(int)
        
        # Distance to boundary
        boundary = gdf.unary_union.boundary
        features['dist_to_water'] = centroid.distance(boundary)
        features['elevation'] = gdf.get('elevation', 10.0)
        
        # ===== SOCIOECONOMIC CONTEXT (7 features) =====
        features['assessed_total'] = gdf['assessed_total'].fillna(0)
        features['assessed_land'] = gdf['assessed_land'].fillna(0)
        features['assessed_building'] = (
            gdf['assessed_total'].fillna(0) - gdf['assessed_land'].fillna(0)
        )
        features['exemption_value'] = gdf.get('exemption_value', 0).fillna(0)
        
        # Census placeholders
        features['median_income'] = 75000
        features['population_density'] = 1000
        features['pct_rental'] = 0.5
        
        # ===== REGULATORY CONSTRAINTS (5 features) =====
        features['historic_district'] = gdf.get('historic_district', 0).fillna(0).astype(int)
        features['landmark'] = gdf.get('landmark', 0).fillna(0).astype(int)
        features['special_district'] = gdf.get('special_district', 0).fillna(0).astype(int)
        features['max_height_ft'] = gdf['num_floors'].fillna(10) * 12
        features['setback_required'] = gdf['zone_district'].str.contains('R', na=False).astype(int)
        
        # ===== DERIVED FEATURES (8 features) =====
        features['age'] = 2024 - features['year_built']
        features['years_since_renovation'] = 2024 - features['year_altered']
        features['far_utilization'] = (
            features['current_far'] / features['max_far'].replace(0, 1)
        ).clip(0, 1)
        features['development_potential'] = (
            features['max_far'] - features['current_far']
        ).clip(0)
        
        features['value_per_sqft'] = (
            features['assessed_total'] / features['area_sqft'].replace(0, 1)
        ).replace([np.inf, -np.inf], 0)
        
        units_res = gdf['units_residential'].fillna(0)
        units_total = gdf['units_total'].fillna(0)
        
        features['units_per_acre'] = units_res / features['area_sqft'] * 43560
        features['jobs_density'] = (units_total - units_res) / features['area_sqft'] * 43560
        features['land_use_mix'] = (
            (gdf.get('comm_far', 0) > 0) & (gdf.get('resid_far', 0) > 0)
        ).astype(int)
        
        # Merge land use dummies
        features = pd.concat([features, landuse_dummies], axis=1)
        
        print(f"Generated {len(features.columns)} features for {len(features)} parcels")
        return features
    
    def normalize_features(self) -> pd.DataFrame:
        """Apply Z-score normalization to features."""
        if self.features is None:
            raise ValueError("Features not computed. Call compute_node_features first.")
        
        scaler = StandardScaler()
        self.features_normalized = pd.DataFrame(
            scaler.fit_transform(self.features),
            columns=self.features.columns,
            index=self.features.index
        )
        
        print(f"Normalized features: mean={self.features_normalized.mean().mean():.4f}, "
              f"std={self.features_normalized.std().mean():.4f}")
        return self.features_normalized
    
    def load_data(self) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Main loading pipeline.
        
        Returns:
            Tuple of (parcels_gdf, features_df)
        """
        print("=" * 60)
        print(f"{self.config.display_name.upper()} PARCEL DATA LOADER")
        print("=" * 60)
        
        # Load parcels
        self.parcels_gdf = self.download_parcels()
        self.parcels_gdf = self.standardize_columns(self.parcels_gdf)
        print(f"Loaded {len(self.parcels_gdf)} parcels")
        
        # Load zoning
        self.zoning_gdf = self.download_zoning()
        print(f"Loaded {len(self.zoning_gdf)} zoning districts")
        
        # Compute features
        self.features = self.compute_node_features(self.parcels_gdf)
        
        # Normalize
        self.normalize_features()
        
        print("=" * 60)
        print("DATA LOADING COMPLETE")
        print("=" * 60)
        
        return self.parcels_gdf, self.features


class ManhattanDataLoader(CityDataLoader):
    """
    Data loader for Manhattan, NYC using MapPLUTO dataset.
    
    Data source: NYC Open Data
    - MapPLUTO: https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks
    - Zoning: https://data.cityofnewyork.us/City-Government/Zoning-Districts/7823-25i9
    """
    
    PLUTO_URL = "https://data.cityofnewyork.us/api/geospatial/64uk-42ks?method=export&format=Shapefile"
    ZONING_URL = "https://data.cityofnewyork.us/api/geospatial/7823-25i9?method=export&format=Shapefile"
    
    def __init__(self, data_dir: Optional[Path] = None, cache: bool = True):
        super().__init__('manhattan', data_dir, cache)
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Map MapPLUTO columns to standard schema."""
        return {
            'BBL': 'parcel_id',
            'Address': 'address',
            'LotArea': 'lot_area_sqft',
            'BldgArea': 'bldg_area_sqft',
            'NumFloors': 'num_floors',
            'YearBuilt': 'year_built',
            'YearAlter1': 'year_altered',
            'LandUse': 'land_use',
            'ZoneDist1': 'zone_district',
            'BuiltFAR': 'built_far',
            'ResidFAR': 'resid_far',
            'CommFAR': 'comm_far',
            'FacilFAR': 'facil_far',
            'AssessTot': 'assessed_total',
            'AssessLand': 'assessed_land',
            'UnitsRes': 'units_residential',
            'UnitsTotal': 'units_total',
            'BldgClass': 'bldg_class',
            'HistDist': 'historic_district',
            'Landmark': 'landmark',
            'SplitZone': 'special_district',
            'ExemptTot': 'exemption_value',
        }
    
    def download_parcels(self) -> gpd.GeoDataFrame:
        """Download MapPLUTO data for Manhattan."""
        cache_file = self.data_dir / 'pluto_manhattan.geojson'
        
        if self.cache and cache_file.exists():
            print("Loading cached MapPLUTO data...")
            return gpd.read_file(cache_file)
        
        print("Downloading MapPLUTO data from NYC Open Data...")
        response = requests.get(self.PLUTO_URL, stream=True, timeout=120)
        
        with ZipFile(BytesIO(response.content)) as z:
            z.extractall(self.data_dir / 'pluto_raw')
        
        shp_file = list((self.data_dir / 'pluto_raw').glob('*.shp'))[0]
        gdf = gpd.read_file(shp_file)
        
        # Filter to Manhattan (BoroCode == 1)
        manhattan = gdf[gdf['BoroCode'] == '1'].copy()
        
        # Compute max_far before standardization
        manhattan['max_far'] = (
            manhattan['ResidFAR'].fillna(0) + 
            manhattan['CommFAR'].fillna(0) + 
            manhattan['FacilFAR'].fillna(0)
        )
        
        # Save cache
        manhattan.to_file(cache_file, driver='GeoJSON')
        print(f"Saved {len(manhattan)} Manhattan parcels to cache")
        
        return manhattan
    
    def download_zoning(self) -> gpd.GeoDataFrame:
        """Download zoning district data."""
        cache_file = self.data_dir / 'zoning_manhattan.geojson'
        
        if self.cache and cache_file.exists():
            print("Loading cached zoning data...")
            return gpd.read_file(cache_file)
        
        print("Downloading zoning data...")
        response = requests.get(self.ZONING_URL, stream=True, timeout=120)
        
        with ZipFile(BytesIO(response.content)) as z:
            z.extractall(self.data_dir / 'zoning_raw')
        
        shp_file = list((self.data_dir / 'zoning_raw').glob('*.shp'))[0]
        zoning = gpd.read_file(shp_file)
        
        # Filter to Manhattan bounds
        if self.parcels_gdf is not None:
            bounds = self.parcels_gdf.total_bounds
            zoning = zoning.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        
        zoning.to_file(cache_file, driver='GeoJSON')
        return zoning


class ChicagoDataLoader(CityDataLoader):
    """
    Data loader for Chicago, IL using Cook County Assessor data.
    
    Data source: Cook County Data Portal
    """
    
    def __init__(self, data_dir: Optional[Path] = None, cache: bool = True):
        super().__init__('chicago', data_dir, cache)
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Map Chicago columns to standard schema."""
        return {
            'PIN': 'parcel_id',
            'property_address': 'address',
            'land_area': 'lot_area_sqft',
            'building_area': 'bldg_area_sqft',
            'stories': 'num_floors',
            'year_built': 'year_built',
            'class': 'land_use',
            'zoning': 'zone_district',
            'far': 'built_far',
            'max_far': 'max_far',
            'assessed_value': 'assessed_total',
            'land_value': 'assessed_land',
            'residential_units': 'units_residential',
            'total_units': 'units_total',
        }
    
    def download_parcels(self) -> gpd.GeoDataFrame:
        """Download Chicago parcel data."""
        cache_file = self.data_dir / 'parcels_chicago.geojson'
        
        if self.cache and cache_file.exists():
            print("Loading cached Chicago parcel data...")
            return gpd.read_file(cache_file)
        
        # TODO: Implement Chicago data download
        # For now, return empty GeoDataFrame with expected schema
        print("WARNING: Chicago data loader not fully implemented. Using placeholder.")
        return gpd.GeoDataFrame(columns=list(self.get_column_mapping().keys()))
    
    def download_zoning(self) -> gpd.GeoDataFrame:
        """Download Chicago zoning data."""
        cache_file = self.data_dir / 'zoning_chicago.geojson'
        
        if self.cache and cache_file.exists():
            return gpd.read_file(cache_file)
        
        # TODO: Implement Chicago zoning download
        return gpd.GeoDataFrame()


class LADataLoader(CityDataLoader):
    """
    Data loader for Los Angeles, CA using LA County Open Data.
    
    Data source: LA County GIS Data Portal
    """
    
    def __init__(self, data_dir: Optional[Path] = None, cache: bool = True):
        super().__init__('la', data_dir, cache)
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Map LA columns to standard schema."""
        return {
            'APN': 'parcel_id',
            'SitusAddress': 'address',
            'LandArea': 'lot_area_sqft',
            'BuildingArea': 'bldg_area_sqft',
            'Stories': 'num_floors',
            'YearBuilt': 'year_built',
            'UseCode': 'land_use',
            'Zoning': 'zone_district',
            'FAR': 'built_far',
            'MaxFAR': 'max_far',
            'AssessedValue': 'assessed_total',
            'LandValue': 'assessed_land',
            'ResUnits': 'units_residential',
            'TotalUnits': 'units_total',
        }
    
    def download_parcels(self) -> gpd.GeoDataFrame:
        """Download LA parcel data."""
        cache_file = self.data_dir / 'parcels_la.geojson'
        
        if self.cache and cache_file.exists():
            print("Loading cached LA parcel data...")
            return gpd.read_file(cache_file)
        
        # TODO: Implement LA data download
        print("WARNING: LA data loader not fully implemented. Using placeholder.")
        return gpd.GeoDataFrame(columns=list(self.get_column_mapping().keys()))
    
    def download_zoning(self) -> gpd.GeoDataFrame:
        """Download LA zoning data."""
        cache_file = self.data_dir / 'zoning_la.geojson'
        
        if self.cache and cache_file.exists():
            return gpd.read_file(cache_file)
        
        # TODO: Implement LA zoning download
        return gpd.GeoDataFrame()


class BostonDataLoader(CityDataLoader):
    """
    Data loader for Boston, MA using Boston GIS Open Data.
    
    Data source: Boston GIS / Analyze Boston
    """
    
    def __init__(self, data_dir: Optional[Path] = None, cache: bool = True):
        super().__init__('boston', data_dir, cache)
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Map Boston columns to standard schema."""
        return {
            'parcel_num': 'parcel_id',
            'st_name': 'address',
            'lot_size': 'lot_area_sqft',
            'gross_area': 'bldg_area_sqft',
            'num_floors': 'num_floors',
            'yr_built': 'year_built',
            'lu': 'land_use',
            'zoning': 'zone_district',
            'far': 'built_far',
            'max_far': 'max_far',
            'av_total': 'assessed_total',
            'av_land': 'assessed_land',
            'res_units': 'units_residential',
            'total_units': 'units_total',
        }
    
    def download_parcels(self) -> gpd.GeoDataFrame:
        """Download Boston parcel data."""
        cache_file = self.data_dir / 'parcels_boston.geojson'
        
        if self.cache and cache_file.exists():
            print("Loading cached Boston parcel data...")
            return gpd.read_file(cache_file)
        
        # TODO: Implement Boston data download
        print("WARNING: Boston data loader not fully implemented. Using placeholder.")
        return gpd.GeoDataFrame(columns=list(self.get_column_mapping().keys()))
    
    def download_zoning(self) -> gpd.GeoDataFrame:
        """Download Boston zoning data."""
        cache_file = self.data_dir / 'zoning_boston.geojson'
        
        if self.cache and cache_file.exists():
            return gpd.read_file(cache_file)
        
        # TODO: Implement Boston zoning download
        return gpd.GeoDataFrame()


def get_data_loader(city: str, **kwargs) -> CityDataLoader:
    """
    Factory function to get the appropriate data loader for a city.
    
    Args:
        city: City identifier ('manhattan', 'chicago', 'la', 'boston')
        **kwargs: Additional arguments passed to loader constructor
        
    Returns:
        CityDataLoader instance for the requested city
        
    Raises:
        ValueError: If city is not supported
    """
    loaders = {
        'manhattan': ManhattanDataLoader,
        'chicago': ChicagoDataLoader,
        'la': LADataLoader,
        'boston': BostonDataLoader,
    }
    
    if city not in loaders:
        raise ValueError(
            f"Unsupported city: {city}. Available: {list(loaders.keys())}"
        )
    
    return loaders[city](**kwargs)


# Example usage
if __name__ == "__main__":
    # Load Manhattan data
    loader = get_data_loader('manhattan')
    gdf, features = loader.load_data()
    
    print("\nFeature Summary:")
    print(features.describe())
    
    print("\nSample parcels:")
    print(gdf[['address', 'zone_district', 'land_use', 'bldg_area_sqft', 'num_floors']].head(10))
