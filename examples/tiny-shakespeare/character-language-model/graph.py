"""Build deterministic synthetic or MaleCNS dense induced graphs."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather


def build_graph(source: str, neurons: int, threshold: int, seed: int):
    rng = np.random.default_rng(seed)
    if source == "synthetic":
        src = rng.integers(0, neurons, 8 * neurons)
        dst = rng.integers(0, neurons, 8 * neurons)
        val = rng.integers(1, 20, 8 * neurons).astype(np.float32)
        ids = np.arange(neurons)
    else:
        table = feather.read_table(Path(source), memory_map=True)
        table = table.select(("body_pre", "body_post", "weight")).filter(
            pc.greater_equal(table["weight"], threshold))
        raw_src = table["body_pre"].to_numpy().astype(np.int64, copy=False)
        raw_dst = table["body_post"].to_numpy().astype(np.int64, copy=False)
        raw_val = table["weight"].to_numpy().astype(np.float32, copy=False)
        all_ids, inv = np.unique(np.concatenate((raw_src, raw_dst)), return_inverse=True)
        score = np.bincount(inv, weights=np.concatenate((raw_val, raw_val)))
        chosen = np.lexsort((rng.random(len(all_ids)), -score))[:neurons]
        ids = np.sort(all_ids[chosen])
        src_i, dst_i = np.searchsorted(ids, raw_src), np.searchsorted(ids, raw_dst)
        safe_s, safe_d = np.minimum(src_i, len(ids)-1), np.minimum(dst_i, len(ids)-1)
        keep = (src_i < len(ids)) & (dst_i < len(ids)) & (ids[safe_s] == raw_src) & (ids[safe_d] == raw_dst)
        src, dst, val = src_i[keep], dst_i[keep], raw_val[keep]
    signs = rng.choice(np.array([-1., 1.], np.float32), len(ids), p=[.25, .75])
    val = np.log1p(val) * signs[src]
    incoming = np.bincount(dst, weights=np.abs(val), minlength=len(ids)).astype(np.float32)
    val /= np.maximum(incoming[dst], 1.)
    adjacency = np.zeros((len(ids), len(ids)), dtype=np.float32)
    np.add.at(adjacency, (dst, src), val)
    return ids, adjacency
