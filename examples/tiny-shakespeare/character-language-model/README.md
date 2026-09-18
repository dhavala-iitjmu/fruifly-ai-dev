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
  --neurons 1024 --iterations 5000 --output runs/malecns
python generate.py --run runs/malecns --prompt $'ROMEO:\n' --characters 500
```

Outputs under `runs/latest/` include the best safetensors weights, graph,
selected node IDs, configuration and validation metrics. The configuration
records `graph` and `backend` so synthetic and MaleCNS runs cannot be confused.

## Model

```text
character -> 32-D embedding -> sensory projection
          -> fixed recurrent connectome core -> 65-way readout
```

Only the embedding, sensory projection and character readout are trained. The
connectome adjacency stays fixed. Validation perplexity and elapsed time are
reported during training.

On the development Apple-silicon machine, a 256-node synthetic-graph run took
about 25 seconds for 5,000 iterations. Its best validation perplexity was 8.65.
These figures are a reproducibility check, not a benchmark against modern
language models.

The verified real-connectome run selected 1,024 nodes from MaleCNS v1.0 and
trained for 5,000 iterations in 96.6 seconds (excluding graph extraction). Its
best validation perplexity was 10.34. The local checkpoint is
`runs/malecns/best.safetensors`; generated artifacts remain Git-ignored.

## Generate interactively

Open `generate_text.ipynb` from this directory after training. It restores the
checkpoint, reports its graph source, and compares multiple temperatures.
