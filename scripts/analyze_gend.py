from __future__ import annotations

import json
from statistics import mean, median

from pathlib import Path


INPUT_PATH = Path(
    "data/processed/gend_predictions.json"
)


def main() -> None:
    data = json.loads(
        INPUT_PATH.read_text()
    )

    predictions = data["predictions"]

    fake_probs = [
        p["fake_probability"]
        for p in predictions
    ]

    real_probs = [
        p["real_probability"]
        for p in predictions
    ]

    fake_frames = [
        p
        for p in predictions
        if p["prediction"] == "fake"
    ]

    real_frames = [
        p
        for p in predictions
        if p["prediction"] == "real"
    ]

    print("Model:", data["model"])
    print("Track:", data["track_id"])
    print("Frames:", len(predictions))

    print()
    print("=== Probability statistics ===")

    print(
        "Mean fake probability:",
        f"{mean(fake_probs):.6f}",
    )

    print(
        "Median fake probability:",
        f"{median(fake_probs):.6f}",
    )

    print(
        "Minimum fake probability:",
        f"{min(fake_probs):.6f}",
    )

    print(
        "Maximum fake probability:",
        f"{max(fake_probs):.6f}",
    )

    print()
    print("=== Frame classifications ===")

    print(
        "Fake frames:",
        len(fake_frames),
    )

    print(
        "Real frames:",
        len(real_frames),
    )

    print(
        "Fake percentage:",
        f"{100 * len(fake_frames) / len(predictions):.2f}%",
    )

    print(
        "Real percentage:",
        f"{100 * len(real_frames) / len(predictions):.2f}%",
    )

    print()
    print("=== Lowest fake probabilities ===")

    lowest = sorted(
        predictions,
        key=lambda p: p["fake_probability"],
    )[:5]

    for p in lowest:
        print(
            f"frame={p['frame_index']:02d} "
            f"time={p['timestamp_seconds']:.1f}s "
            f"fake={p['fake_probability']:.6f} "
            f"real={p['real_probability']:.6f} "
            f"prediction={p['prediction']}"
        )

    print()
    print("=== Highest fake probabilities ===")

    highest = sorted(
        predictions,
        key=lambda p: p["fake_probability"],
        reverse=True,
    )[:5]

    for p in highest:
        print(
            f"frame={p['frame_index']:02d} "
            f"time={p['timestamp_seconds']:.1f}s "
            f"fake={p['fake_probability']:.6f} "
            f"real={p['real_probability']:.6f} "
            f"prediction={p['prediction']}"
        )

    print()
    print("=== Temporal sequence ===")

    for p in predictions:
        print(
            f"{p['timestamp_seconds']:4.1f}s "
            f"{p['fake_probability']:.4f} "
            f"{p['prediction']}"
        )


if __name__ == "__main__":
    main()
