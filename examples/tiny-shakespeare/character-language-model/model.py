"""MLX connectome-constrained character language model."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ConnectomeLM(nn.Module):
    def __init__(self, adjacency, vocab_size: int, embedding_dim: int, steps: int):
        super().__init__()
        self.adjacency = mx.array(adjacency)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input = nn.Linear(embedding_dim, adjacency.shape[0], bias=False)
        self.readout = nn.Linear(adjacency.shape[0], vocab_size)
        self.steps = steps

    def __call__(self, tokens, state=None):
        batch, length = tokens.shape
        if state is None:
            state = mx.zeros((batch, self.adjacency.shape[0]))
        outputs = []
        for t in range(length):
            stimulus = self.input(self.embedding(tokens[:, t]))
            for _ in range(self.steps):
                recurrent = state @ self.adjacency.T
                state = mx.tanh(0.65 * state + 0.8 * recurrent + 0.35 * stimulus)
            outputs.append(self.readout(state))
        return mx.stack(outputs, axis=1), state
