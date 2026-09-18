"""Download and encode the public-domain Tiny Shakespeare corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    raw = args.data_dir / "tinyshakespeare.txt"
    if args.force or not raw.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, raw)
    text = raw.read_text(encoding="utf-8")
    chars = sorted(set(text)); stoi = {ch: i for i, ch in enumerate(chars)}
    encoded = np.fromiter((stoi[ch] for ch in text), dtype=np.uint16)
    n = len(encoded); train_end = int(n * 0.9); val_end = int(n * 0.95)
    np.save(args.data_dir / "train.npy", encoded[:train_end])
    np.save(args.data_dir / "val.npy", encoded[train_end:val_end])
    np.save(args.data_dir / "test.npy", encoded[val_end:])
    meta = {"source": URL, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "characters": chars, "vocab_size": len(chars), "length": n,
            "splits": {"train": train_end, "val": val_end-train_end, "test": n-val_end}}
    (args.data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["splits"] | {"vocab_size": len(chars)}))


if __name__ == "__main__":
    main()
