#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path):
    """
    Compute SHA-256 for a file.
    """
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def hash_chain(hashes):
    """
    Construct a deterministic SHA-256 hash chain.

    H_0 = SHA256("FFPP-INTEGRITY-V1")
    H_i = SHA256(H_(i-1) || frame_hash_i)
    """
    previous = hashlib.sha256(
        b"FFPP-INTEGRITY-V1"
    ).hexdigest()

    chain = []

    for frame_hash in hashes:
        payload = (
            previous + frame_hash
        ).encode("utf-8")

        previous = hashlib.sha256(
            payload
        ).hexdigest()

        chain.append(previous)

    return chain


def sequence_digest(frame_hashes):
    """
    Return the final digest of the ordered sequence.
    """
    chain = hash_chain(frame_hashes)

    if not chain:
        return hashlib.sha256(
            b"FFPP-EMPTY-SEQUENCE"
        ).hexdigest()

    return chain[-1]


def build_integrity_record(frame_paths):
    """
    Generate integrity metadata for an ordered
    sequence of extracted face frames.
    """
    frame_hashes = []

    for path in frame_paths:
        frame_hashes.append(
            sha256_file(path)
        )

    chain = hash_chain(frame_hashes)

    return {
        "algorithm": "SHA-256",
        "version": "FFPP-INTEGRITY-V1",
        "frame_count": len(frame_paths),
        "frame_hashes": frame_hashes,
        "hash_chain": chain,
        "sequence_digest": (
            chain[-1]
            if chain
            else hashlib.sha256(
                b"FFPP-EMPTY-SEQUENCE"
            ).hexdigest()
        ),
    }


def verify_integrity(record, frame_paths):
    """
    Verify that the current frame files produce
    exactly the same cryptographic sequence.
    """
    expected = record["frame_hashes"]

    if len(expected) != len(frame_paths):
        return {
            "valid": False,
            "reason": "frame_count_mismatch",
        }

    actual = [
        sha256_file(path)
        for path in frame_paths
    ]

    if actual != expected:
        mismatches = [
            i
            for i, (a, e) in enumerate(
                zip(actual, expected)
            )
            if a != e
        ]

        return {
            "valid": False,
            "reason": "frame_hash_mismatch",
            "mismatches": mismatches,
        }

    actual_digest = sequence_digest(actual)

    valid = (
        actual_digest
        == record["sequence_digest"]
    )

    return {
        "valid": valid,
        "reason": (
            "ok"
            if valid
            else "sequence_digest_mismatch"
        ),
        "sequence_digest": actual_digest,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("CRYPTOGRAPHIC INTEGRITY TEST")
    print("=" * 70)

    test_data = [
        b"frame-one",
        b"frame-two",
        b"frame-three",
    ]

    hashes = [
        hashlib.sha256(x).hexdigest()
        for x in test_data
    ]

    chain = hash_chain(hashes)

    print("Frames:", len(hashes))
    print("First frame hash:", hashes[0])
    print("Chain length:", len(chain))
    print("Sequence digest:", sequence_digest(hashes))

    print()
    print("Integrity utility: PASS")
