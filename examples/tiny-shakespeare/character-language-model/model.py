"""MLX connectome-constrained character language model."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ConnectomeLM(nn.Module):
    def __init__(self, adjacency, vocab_size: int, embedding_dim: int, steps: int,
                 layers: int = 1, adapter_dim: int = 64, readout_blocks: int = 0):
        super().__init__()
        self.adjacency = mx.array(adjacency)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input = nn.Linear(embedding_dim, adjacency.shape[0], bias=False)
        self.adapters = [
            nn.Sequential(
                nn.Linear(adjacency.shape[0], adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, adjacency.shape[0]),
            )
            for _ in range(layers - 1)
        ]
        self.output_adapters = [
            nn.Sequential(
                nn.Linear(adjacency.shape[0], adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, adjacency.shape[0]),
            )
            for _ in range(readout_blocks)
        ]
        self.readout = nn.Linear(adjacency.shape[0], vocab_size)
        self.steps = steps
        self.layers = layers
        # MLX treats arrays attached to a Module as trainable parameters unless
        # explicitly frozen. The connectome is a structural prior, not a weight
        # to optimize.
        self.freeze(recurse=False, keys="adjacency", strict=True)

    def __call__(self, tokens, state=None):
        batch, length = tokens.shape
        if state is None:
            states = [mx.zeros((batch, self.adjacency.shape[0])) for _ in range(self.layers)]
        elif self.layers == 1:
            states = [state]
        else:
            states = state
        outputs = []
        for t in range(length):
            drive = self.input(self.embedding(tokens[:, t]))
            next_states = []
            for layer in range(self.layers):
                layer_state = states[layer]
                if layer > 0:
                    lower = next_states[layer - 1]
                    drive = lower + 0.1 * self.adapters[layer - 1](lower)
                for _ in range(self.steps):
                    recurrent = layer_state @ self.adjacency.T
                    layer_state = mx.tanh(0.65 * layer_state + 0.8 * recurrent + 0.35 * drive)
                next_states.append(layer_state)
            states = next_states
            features = states[-1]
            for adapter in self.output_adapters:
                features = features + 0.1 * adapter(features)
            outputs.append(self.readout(features))
        final_state = states[0] if self.layers == 1 else states
        return mx.stack(outputs, axis=1), final_state
