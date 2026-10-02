import numpy as np

from pimaluos.core.graph_builder import ALL_EDGE_TYPES, RELATION_NAMES


def test_all_relations_symmetric_no_self_loops(graph):
    data, b = graph
    for et in ALL_EDGE_TYPES:
        ei = data[RELATION_NAMES[et]].edge_index.numpy()
        assert ei.shape[1] > 0, et
        assert (ei[0] != ei[1]).all()
        fwd = set(map(tuple, ei.T))
        assert all((d, s) in fwd for s, d in fwd)
        assert data[RELATION_NAMES[et]].edge_attr.shape == (ei.shape[1], 1)


def test_summary_counts_are_exact(graph):
    data, b = graph
    s = b.summary()
    for et, c in s["edge_types"].items():
        assert c["directed"] == 2 * c["undirected"]
    assert s["total_directed"] == sum(data[r].edge_index.shape[1] for r in data.edge_types)


def test_functional_edges_join_identical_land_use(graph, ds):
    data, _ = graph
    ei = data[RELATION_NAMES["functional_similarity"]].edge_index.numpy()
    lu = ds.gdf["land_use"].values
    assert np.all(lu[ei[0]] == lu[ei[1]])
