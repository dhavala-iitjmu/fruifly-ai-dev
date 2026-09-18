"""MNIST classification with a fixed sparse MaleCNS-connectome reservoir."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pyarrow.compute as pc
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _find_column(names: list[str], candidates: tuple[str, ...]) -> str:
    lower = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f"Could not find one of {candidates}; columns are {names}")


def load_real_graph(path: Path, n: int, threshold: int, seed: int):
    table = feather.read_table(path, memory_map=True)
    names = table.column_names
    pre = _find_column(names, ("body_pre", "bodypre", "source", "pre"))
    post = _find_column(names, ("body_post", "bodypost", "target", "post"))
    weight = _find_column(names, ("weight", "count", "syn_count", "n"))
    # Filter while columns are Arrow-backed/memory-mapped. Converting all 151M
    # rows to NumPy first would need several unnecessary GB of working memory.
    table = table.select((pre, post, weight)).filter(pc.greater_equal(table[weight], threshold))
    src = table[pre].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    dst = table[post].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    val = table[weight].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    # Prefer neurons with the largest strong-edge traffic; jitter breaks ties
    # reproducibly and prevents dependence on Arrow row ordering.
    ids, inverse = np.unique(np.concatenate((src, dst)), return_inverse=True)
    scores = np.bincount(inverse, weights=np.concatenate((val, val)))
    rng = np.random.default_rng(seed)
    rank = np.lexsort((rng.random(ids.size), -scores))[: min(n, ids.size)]
    chosen = np.sort(ids[rank])
    src_i = np.searchsorted(chosen, src)
    dst_i = np.searchsorted(chosen, dst)
    inside = ((src_i < len(chosen)) & (dst_i < len(chosen)) &
              (chosen[np.minimum(src_i, len(chosen)-1)] == src) &
              (chosen[np.minimum(dst_i, len(chosen)-1)] == dst))
    src_i, dst_i, val = src_i[inside], dst_i[inside], val[inside]
    return chosen, src_i, dst_i, val


def synthetic_graph(n: int, seed: int):
    rng = np.random.default_rng(seed)
    edges = max(8 * n, 1)
    src = rng.integers(0, n, edges)
    dst = rng.integers(0, n, edges)
    val = rng.integers(1, 20, edges).astype(np.float32)
    return np.arange(n), src, dst, val


def make_sparse(n: int, src, dst, weights, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], np.float32), size=n, p=[0.25, 0.75])
    weights = np.log1p(weights) * signs[src]
    incoming = np.bincount(dst, weights=np.abs(weights), minlength=n).astype(np.float32)
    weights = weights / np.maximum(incoming[dst], 1.0)
    # sparse.mm computes W @ state, so rows are destinations.
    indices = torch.tensor(np.stack((dst, src)), dtype=torch.long)
    values = torch.tensor(weights, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


class FlyReservoir(nn.Module):
    def __init__(self, graph: torch.Tensor, steps: int, train_input: bool):
        super().__init__()
        self.register_buffer("graph", graph)
        self.steps = steps
        n = graph.shape[0]
        self.input = nn.Linear(28 * 28, n, bias=False)
        self.input.weight.requires_grad_(train_input)
        self.readout = nn.Linear(n, 10)

    def forward(self, images):
        stimulus = self.input(images.flatten(1))
        state = torch.tanh(stimulus)
        for _ in range(self.steps):
            recurrent = torch.sparse.mm(self.graph, state.T).T
            state = torch.tanh(0.65 * state + 0.8 * recurrent + 0.35 * stimulus)
        return self.readout(state)


def loader(train: bool, batch: int, limit: int | None):
    ds = datasets.MNIST("data/mnist", train=train, download=True,
                        transform=transforms.ToTensor())
    if limit is not None:
        ds = Subset(ds, range(min(limit, len(ds))))
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=0)


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval(); correct = total = 0
    for x, y in batches:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum()); total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="synthetic", help="Feather file or 'synthetic'")
    p.add_argument("--neurons", type=int, default=8192)
    p.add_argument("--edge-threshold", type=int, default=5)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--test-limit", type=int)
    p.add_argument("--freeze-input", action="store_true")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("runs/latest"))
    args = p.parse_args()
    seed_everything(args.seed)
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if args.graph == "synthetic":
        ids, src, dst, val = synthetic_graph(args.neurons, args.seed)
    else:
        ids, src, dst, val = load_real_graph(Path(args.graph), args.neurons,
                                              args.edge_threshold, args.seed)
    graph = make_sparse(len(ids), src, dst, val, args.seed).to(device)
    print(f"device={device} neurons={len(ids):,} edges={graph._nnz():,}")
    model = FlyReservoir(graph, args.steps, not args.freeze_input).to(device)
    optimizer = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    train_batches = loader(True, args.batch_size, args.train_limit)
    test_batches = loader(False, args.batch_size, args.test_limit)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "neuron_ids.npy", ids)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); running = seen = 0
        for x, y in tqdm(train_batches, desc=f"epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y); loss.backward(); optimizer.step()
            running += loss.detach().item() * y.numel(); seen += y.numel()
        accuracy = evaluate(model, test_batches, device)
        row = {"epoch": epoch, "loss": running / seen, "test_accuracy": accuracy}
        history.append(row); print(json.dumps(row))
        if accuracy > best:
            best = accuracy
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.output / "best.pt")
    metrics = {"best_test_accuracy": best, "neurons": len(ids),
               "edges": graph._nnz(), "history": history,
               "note": "Connectome topology fixed; this is not a biological brain emulation."}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
