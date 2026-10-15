#!/usr/bin/env python3

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from integrity_features import verify_integrity


FACE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "processed"
    / "faces"
)


MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "metadata"
    / "integrity_train.json"
)


FACE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "metadata"
    / "face_samples_train.json"
)


def get_frame_paths(record):
    """
    Locate the extracted face frames belonging
    to one integrity record.

    The integrity manifest identifies the source
    video. The face extraction manifest contains
    the exact ordered frame paths.
    """

    video_id = Path(record["path"]).stem

    root = (
        FACE_ROOT
        / record["split"]
        / record["label"]
        / record["method"]
        / video_id
    )

    if not root.exists():
        return []

    return sorted(
        root.glob("frame_*.jpg")
    )


def main():

    print("=" * 70)
    print("CRYPTOGRAPHIC INTEGRITY VERIFICATION")
    print("=" * 70)

    with MANIFEST.open() as f:
        integrity_records = json.load(f)

    print("Manifest:", MANIFEST)
    print("Records:", len(integrity_records))
    print()

    valid = 0
    invalid = 0

    for index, record in enumerate(
        integrity_records,
        start=1,
    ):

        frame_paths = get_frame_paths(record)

        result = verify_integrity(
            record["integrity"],
            frame_paths,
        )

        if result["valid"]:
            valid += 1
            status = "PASS"
        else:
            invalid += 1
            status = "FAIL"

        print(
            f"[{index:04d}/{len(integrity_records):04d}] "
            f"{status:4s} "
            f"{record['label']:5s} "
            f"{record['method']:20s} "
            f"frames={len(frame_paths)}"
        )

        if not result["valid"]:
            print(
                "    Reason:",
                result.get("reason"),
            )

            if "mismatches" in result:
                print(
                    "    Frame mismatches:",
                    result["mismatches"],
                )

    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("Valid:", valid)
    print("Invalid:", invalid)

    if invalid != 0:
        raise RuntimeError(
            "Integrity verification failed."
        )

    print()
    print("Integrity verification: PASS")


if __name__ == "__main__":
    main()
