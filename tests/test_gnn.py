from pimaluos.models.gnn import ParcelGNN, pretrain_gnn


def test_forward_shapes(graph):
    data, _ = graph
    m = ParcelGNN(data["parcel"].x.shape[1], list(data.edge_types), hidden_channels=32, embed_dim=16, heads=4)
    z = m.get_embeddings(data)
    assert z.shape == (data["parcel"].num_nodes, 16)
    assert set(m.relation_weights()["l1"]) == {r[1] for r in data.edge_types}


def test_pretraining_beats_mean_predictor(graph):
    data, _ = graph
    m = ParcelGNN(data["parcel"].x.shape[1], list(data.edge_types), hidden_channels=64, embed_dim=32)
    h = pretrain_gnn(m, data, epochs=60, seed=0, patience=60)
    assert h["n_val_nodes"] > 0 and len(h["val_loss"]) == 60
    assert h["best_val_loss"] < h["mean_predictor_val_loss"]


def test_no_relation_model_runs(graph):
    data, _ = graph
    m = ParcelGNN(data["parcel"].x.shape[1], [], hidden_channels=32, embed_dim=16)
    assert m.get_embeddings(data).shape[1] == 16
