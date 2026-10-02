import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from pimaluos.config.settings import get_city_config
from pimaluos.core.data_loader import _normalise_land_use, standardise_parcels, transform_features
from pimaluos.core.graph_builder import parse_street_name

MAPPING = get_city_config("manhattan").column_mapping


def _raw(**over):
    row = dict(geometry=box(0, 0, 100, 50), BBL=1, Address="10 WEST 4 STREET", LotArea=5000, BldgArea=10000,
               NumFloors=4, LandUse=3, ZoneDist1="R7A", ResidFAR=4.0, CommFAR=0.0, FacilFAR=4.0,
               BuiltFAR=2.0, AssessTot=1e6, AssessLand=4e5)
    row.update(over)
    return gpd.GeoDataFrame([row], crs="EPSG:2263")


def test_max_far_is_max_not_sum():
    g = standardise_parcels(_raw(ResidFAR=6.02, CommFAR=2.0, FacilFAR=6.5), MAPPING, "EPSG:2263")
    assert np.isclose(g["max_far"].iloc[0], 6.5)


def test_lot_area_uses_assessor_value_and_land_use_codes():
    g = standardise_parcels(_raw(), MAPPING, "EPSG:2263")
    assert g["lot_area_sqft"].iloc[0] == 5000
    assert g["land_use"].iloc[0] == "03"
    assert list(_normalise_land_use(pd.Series([1, "5", None, 11.0]))) == ["01", "05", "11", "11"]


def test_transform_drops_constants_and_standardises():
    f = pd.DataFrame({"a": [1.0, 2, 3, 4], "const": [5.0] * 4, "lot_area_sqft": [10.0, 100, 1000, 10000]})
    t = transform_features(f)
    assert "const" not in t.columns
    assert np.allclose(t.mean(), 0, atol=1e-9) and np.allclose(t.std(ddof=0), 1)


def test_synthetic_dataset(ds):
    assert ds.meta["synthetic"] is True
    assert ds.features.shape[0] == len(ds.gdf)
    assert ds.features.notna().all().all()
    caps = ds.gdf[["max_resid_far", "max_comm_far", "max_facil_far"]].max(axis=1)
    assert np.allclose(ds.gdf["max_far"], caps)


def test_parse_street_name_keeps_numbers():
    assert parse_street_name("123 WEST 45 STREET") == "WEST 45 STREET"
    assert parse_street_name("500 WEST 110 STREET") == "WEST 110 STREET"
    assert parse_street_name("12-14 E 4TH ST") == "EAST 4 STREET"
    assert parse_street_name("") is None


def test_manufacturing_cap_included_affordable_excluded_by_default():
    raw = _raw(ResidFAR=0.0, CommFAR=2.0, FacilFAR=4.8, ManuFAR=5.0, AffResFAR=7.2)
    g = standardise_parcels(raw, MAPPING, "EPSG:2263")
    assert np.isclose(g["max_far"].iloc[0], 5.0)
    g2 = standardise_parcels(raw, MAPPING, "EPSG:2263", include_affordable_far=True)
    assert np.isclose(g2["max_far"].iloc[0], 7.2)
