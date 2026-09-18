import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from graph import build_graph
from model import ConnectomeLM


def test_graph_is_deterministic():
    ids1, graph1 = build_graph("synthetic", 32, 5, 7)
    ids2, graph2 = build_graph("synthetic", 32, 5, 7)
    np.testing.assert_array_equal(ids1, ids2)
    np.testing.assert_allclose(graph1, graph2)


def test_connectome_adjacency_is_frozen():
    _, graph = build_graph("synthetic", 32, 5, 7)
    model = ConnectomeLM(graph, vocab_size=11, embedding_dim=8, steps=1)
    assert "adjacency" not in model.trainable_parameters()
    assert "adjacency" in model.parameters()


def test_forward_shape():
    _, graph = build_graph("synthetic", 32, 5, 7)
    model = ConnectomeLM(graph, vocab_size=11, embedding_dim=8, steps=1)
    logits, state = model(mx.array([[1, 2, 3], [3, 2, 1]]))
    mx.eval(logits, state)
    assert logits.shape == (2, 3, 11)
    assert state.shape == (2, 32)


def test_stacked_forward_shape():
    _, graph = build_graph("synthetic", 32, 5, 7)
    model = ConnectomeLM(graph, vocab_size=11, embedding_dim=8, steps=1,
                         layers=2, adapter_dim=4)
    logits, states = model(mx.array([[1, 2, 3], [3, 2, 1]]))
    mx.eval(logits, states)
    assert logits.shape == (2, 3, 11)
    assert len(states) == 2
    assert states[0].shape == states[1].shape == (2, 32)


def test_nonlinear_readout_block_shape():
    _, graph = build_graph("synthetic", 32, 5, 7)
    model = ConnectomeLM(graph, vocab_size=11, embedding_dim=8, steps=1,
                         adapter_dim=4, readout_blocks=1)
    logits, state = model(mx.array([[1, 2, 3], [3, 2, 1]]))
    mx.eval(logits, state)
    assert logits.shape == (2, 3, 11)


def test_metadata_round_trip(tmp_path: Path):
    meta = {"characters": ["a", "b", "c"], "vocab_size": 3}
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta))
    assert json.loads(path.read_text()) == meta
