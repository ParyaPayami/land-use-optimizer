import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def ds():
    from pimaluos.core import SyntheticCityLoader

    return SyntheticCityLoader(n_blocks_x=3, n_blocks_y=4, lots_per_block=8, seed=1).load()


@pytest.fixture(scope="session")
def graph(ds):
    from pimaluos.core import ParcelGraphBuilder

    b = ParcelGraphBuilder(ds.gdf, ds.features)
    return b.build(), b


@pytest.fixture(scope="session")
def cap(ds):
    from pimaluos.physics import CapacityModel

    return CapacityModel(ds.gdf)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)
