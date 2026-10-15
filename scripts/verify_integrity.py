#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "metadata"
    / "integrity_manifest.json"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Verify FF++ files against SHA-256 manifest."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    args = parser.parse_args()

    with args.manifest.open() as f:
        manifest = json.load(f)

    records = manifest["records"]

    print("FF++ integrity verification")
    print("=" * 70)

    failures = []

    for i, record in enumerate(records, start=1):
        path = PROJECT_ROOT / record["path"]

        if not path.exists():
            failures.append(
                {
                    "path": record["path"],
                    "reason": "missing",
                }
            )
            continue

        actual_size = path.stat().st_size

        if actual_size != record["size_bytes"]:
            failures.append(
                {
                    "path": record["path"],
                    "reason": "size_mismatch",
                }
            )
            continue

        actual_hash = sha256_file(path)

        if actual_hash != record["sha256"]:
            failures.append(
                {
                    "path": record["path"],
                    "reason": "sha256_mismatch",
                }
            )

        if i % 100 == 0 or i == len(records):
            print(f"[{i:04d}/{len(records):04d}]")

    print()
    print("Checked:", len(records))
    print("Failures:", len(failures))

    if failures:
        print()
        for failure in failures[:20]:
            print(failure)

        raise SystemExit(1)

    print("INTEGRITY: PASS")


if __name__ == "__main__":
    main()
