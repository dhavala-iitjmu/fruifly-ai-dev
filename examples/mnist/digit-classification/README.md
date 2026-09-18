# MNIST digit classification through the MaleCNS fruit-fly connectome

This experiment uses the released **MaleCNS v1.0 connection graph** as a fixed
sparse recurrent reservoir. MNIST pixels stimulate reservoir neurons and a
small, trainable linear layer reads out one of ten digits.

This is deliberately described as *connectome-constrained reservoir
computing*, not as training a scanned fly brain. MaleCNS is a static wiring
diagram (neuron IDs and synapse counts), not a released executable brain or a
set of learned weights. The activation dynamics, image-to-neuron mapping, and
classifier below are engineering choices.

## Relationship to Haltere

[Haltere](https://github.com/skulitom/haltere) is the useful reference for this
style of experiment. It builds a 30,000-neuron flight circuit from MaleCNS,
uses neurotransmitter predictions to fix connection signs, injects telemetry
into annotated sensory populations, and reads controls from annotated wing
motor populations. It then trains connection and neuron dynamics while
regularizing them toward the connectome prior.

This MNIST project is intentionally smaller: it uses the same official
MaleCNS connection table but selects a generic high-strength subgraph, maps
pixels through a learned encoder, keeps recurrent graph weights fixed, and
learns a ten-class readout. Consequently it demonstrates connectome-constrained
classification; it does not reuse Haltere's drone checkpoint or claim that a
biological fly recognizes handwritten digits.

## Quick start

Python 3.10+ is recommended.
Run the shared setup once from the repository root, then enter this example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd examples/mnist/digit-classification

# End-to-end smoke test; no large download and no network needed.
python train.py --graph synthetic --neurons 1024 --steps 2 \
  --epochs 1 --train-limit 2000 --test-limit 500
```

## Run with the real MaleCNS graph

Download the official 1.1 GB aggregated connection table:

```bash
python ../../../tools/download_connectome.py \
  --output ../../../data/connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
python train.py \
  --graph ../../../data/connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --neurons 1024 --steps 3 --epochs 5 --output runs/malecns
```

The default run chooses a deterministic, high-strength 8,192-node subgraph,
which is practical on a laptop with several GB of free memory. It retains only
edges whose two endpoints are in that set, normalizes incoming weights, applies
a fixed random sign to each source (the connectivity table itself contains
counts, not synapse signs), and trains only the input projection and linear
readout. Some nodes in the complete flat table may be untraced segments, so
"node" is more precise here than claiming every selected ID is a curated neuron.

For a larger experiment (more RAM and compute):

```bash
python train.py --graph data/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --neurons 32768 --steps 4 --epochs 10 --device cuda
```

Useful flags:

- `--freeze-input`: train only the final linear classifier.
- `--neurons`: reservoir size. Selection is reproducible for a given seed.
- `--edge-threshold`: discard weak aggregate connections before selection.
- `--train-limit` / `--test-limit`: shorter experiments.
- `--device mps`: Apple Silicon acceleration when supported by sparse ops;
  use `cpu` if the installed PyTorch build rejects sparse MPS operations.

Artifacts are written under `runs/`: the best checkpoint, metrics JSON, and
the exact selected neuron IDs. MNIST is downloaded by torchvision into
`data/mnist` on first use.

After training, open `classify_digits.ipynb` with Jupyter to restore the saved
model, inspect predictions, measure test accuracy, and classify your own image.
The notebook now defaults to the verified real-connectome checkpoint at
`runs/malecns/best.pt`.

The verified run used the same 1,024 selected MaleCNS IDs and 27,709 retained
connections as the Tiny Shakespeare experiment. After five epochs it reached
97.15% accuracy on the complete 10,000-image MNIST test set. Generated data
and checkpoints remain Git-ignored.

## Data provenance

MaleCNS v1.0 was released by HHMI Janelia, Cambridge/MRC LMB collaborators,
and Google Research under CC BY 4.0. The shared download tool fetches the unmodified
aggregate weights table from the official `flyem-male-cns` Google Cloud
Storage bucket. See <https://male-cns.janelia.org/download/>.
