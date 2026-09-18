# Fruit-fly AI experiments

Experiments that apply the Janelia MaleCNS v1.0 fruit-fly connectome to
machine-learning tasks. The repository is organized by **dataset**, then by
**task**, so additional experiments can be added without mixing their models,
checkpoints, or notebooks.

## Examples

| Dataset | Task | Location |
|---|---|---|
| MNIST | Digit classification | [`examples/mnist/digit-classification`](examples/mnist/digit-classification) |

## Repository layout

```text
examples/
  mnist/
    digit-classification/
      README.md
      train.py
      classify_digits.ipynb
tools/
  download_connectome.py
requirements.txt
```

Each example owns its training code, notebooks, `data/` directory, and `runs/`
directory. Generated data and checkpoints are intentionally ignored by Git.
Shared utilities belong under `tools/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then enter an example directory and follow its README:

```bash
cd examples/mnist/digit-classification
```

## Scientific scope

MaleCNS is a static connectome—a wiring diagram—not a pretrained artificial
neural network or a complete executable brain. Every example must document
which dynamics, input mapping, output mapping, and trainable parameters it
adds to the biological connectivity data.

MaleCNS v1.0 was released by HHMI Janelia, Cambridge/MRC LMB collaborators,
and Google Research under CC BY 4.0. See the
[official dataset page](https://male-cns.janelia.org/download/).
