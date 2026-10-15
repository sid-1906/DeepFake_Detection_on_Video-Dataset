#!/usr/bin/env python3

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from integrity_features import build_integrity_record


INPUT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "metadata"
    / "face_samples_train.json"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "metadata"
    / "integrity_train.json"
)


def main():
    with INPUT_MANIFEST.open() as f:
        records = json.load(f)

    output = []

    print("=" * 70)
    print("BUILDING CRYPTOGRAPHIC INTEGRITY MANIFEST")
    print("=" * 70)

    for index, record in enumerate(records, start=1):

        frame_paths = [
            PROJECT_ROOT / frame["image"]
            for frame in record["frames"]
        ]

        integrity = build_integrity_record(
            frame_paths
        )

        output.append({
            "path": record["path"],
            "split": record["split"],
            "label": record["label"],
            "method": record["method"],
            "source_group": record["source_group"],
            "integrity": integrity,
        })

        print(
            f"[{index:04d}/{len(records):04d}] "
            f"{record['label']:4s} "
            f"{record['method']:18s} "
            f"frames={len(frame_paths)}"
        )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_MANIFEST.open("w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("INTEGRITY MANIFEST COMPLETE")
    print("=" * 70)
    print("Videos:", len(output))
    print("Output:", OUTPUT_MANIFEST)


if __name__ == "__main__":
    main()
