"""Download and verify the official MaleCNS v1.0 aggregate connectivity."""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from tqdm import tqdm

URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
    "flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
EXPECTED_SIZE = 1_051_241_946
EXPECTED_SHA256 = "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"


class Progress(tqdm):
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data") / URL.rsplit("/", 1)[-1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        print(f"Already present: {args.output} ({args.output.stat().st_size:,} bytes)")
        digest = sha256(args.output)
        print(f"SHA-256: {digest}")
        if args.output.stat().st_size != EXPECTED_SIZE or digest != EXPECTED_SHA256:
            raise SystemExit("Existing file does not match the official MaleCNS v1.0 artifact")
        return
    partial = args.output.with_suffix(args.output.suffix + ".part")
    with Progress(unit="B", unit_scale=True, miniters=1, desc="MaleCNS") as bar:
        urllib.request.urlretrieve(URL, partial, reporthook=bar.update_to)
    partial.replace(args.output)
    print(f"Saved {args.output} ({args.output.stat().st_size:,} bytes)")
    digest = sha256(args.output)
    print(f"SHA-256: {digest}")
    if args.output.stat().st_size != EXPECTED_SIZE or digest != EXPECTED_SHA256:
        raise SystemExit("Downloaded file failed size/SHA-256 verification")


if __name__ == "__main__":
    main()
