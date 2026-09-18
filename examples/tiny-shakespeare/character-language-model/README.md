# Tiny Shakespeare character language model

A minimal character-level language model whose recurrent core is constrained
by a fixed fruit-fly connectome graph. It predicts the next character in Tiny
Shakespeare and generates text one character at a time.

This is an engineering experiment, not evidence that a biological fly can
process language. Character embeddings, rate dynamics, recurrence timing and
the vocabulary readout are artificial choices.

## Why MLX

The default backend is [MLX](https://github.com/ml-explore/mlx), which runs
natively on Apple silicon and uses unified memory. The selected 256–1,024-node
induced graph is represented as a dense, fixed adjacency matrix; the full
166,000-node connectome is never densified.

## Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd examples/tiny-shakespeare/character-language-model

python prepare_data.py
python train.py --graph synthetic --neurons 256 --iterations 500
python generate.py --prompt $'ROMEO:\n' --characters 500
```

Use the official MaleCNS connectivity:

```bash
python ../../../tools/download_connectome.py \
  --output ../../../data/connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
python train.py \
  --graph ../../../data/connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --neurons 1024 --adapter-dim 128 --readout-blocks 1 \
  --iterations 5000 --output runs/malecns-frozen-adapter
python generate.py --run runs/malecns-frozen-adapter \
  --prompt $'ROMEO:\n' --characters 500
```

The selected `--output` directory contains the best safetensors weights, graph,
selected node IDs, configuration and validation metrics. The configuration
records `graph` and `backend` so synthetic and MaleCNS runs cannot be confused.

## Model

```text
character -> 32-D embedding -> sensory projection
          -> fixed recurrent connectome core
          -> trainable residual GELU adapter -> 65-way readout
```

The adapter is a low-rank `1024 -> 128 -> 1024` block with GELU nonlinearity
and a residual connection. The embedding, sensory projection, adapter and
character readout are trained. The connectome adjacency is explicitly frozen
with MLX's `Module.freeze`; a regression test verifies it is absent from
`trainable_parameters()`.

The model remains causal. A bidirectional RNN was deliberately not used because
it would expose future characters while predicting the next character and
would therefore invalidate autoregressive generation.

## Verified results

Both runs use the same explicitly frozen 1,024-node MaleCNS graph with 27,709
connections and the same validation split:

| Model | Best validation perplexity | MLX training time |
|---|---:|---:|
| Fixed MaleCNS core, linear readout | 6.56 | 61.2 s |
| Fixed MaleCNS core + residual GELU adapter | **6.35** | 89.7 s |

The nonlinear adapter reduces perplexity by 3.2%. The preferred local
checkpoint is `runs/malecns-frozen-adapter/best.safetensors`. Generated
artifacts remain Git-ignored; these figures are reproducibility checks, not
benchmarks against modern language models.

## Generate interactively

Open `generate_text.ipynb` from this directory after training. It restores the
checkpoint, reports its graph source, and compares multiple temperatures.
