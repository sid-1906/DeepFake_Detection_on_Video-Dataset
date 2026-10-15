from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median


INPUT_PATH = Path(
    "data/processed/gend_predictions.json"
)

OUTPUT_PATH = Path(
    "data/processed/video_result.json"
)


def percentile(values, p):
    values = sorted(values)

    if not values:
        raise ValueError(
            "Cannot calculate percentile of empty list."
        )

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = position - lower

    return (
        values[lower]
        + fraction
        * (values[upper] - values[lower])
    )


def longest_run(predictions, label):
    longest = 0
    current = 0

    for prediction in predictions:
        if prediction == label:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return longest


def main():
    data = json.loads(
        INPUT_PATH.read_text()
    )

    predictions = data["predictions"]

    if not predictions:
        raise RuntimeError(
            "No predictions found."
        )

    fake_probs = [
        p["fake_probability"]
        for p in predictions
    ]

    fake_count = sum(
        p["prediction"] == "fake"
        for p in predictions
    )

    real_count = sum(
        p["prediction"] == "real"
        for p in predictions
    )

    labels = [
        p["prediction"]
        for p in predictions
    ]

    result = {
        "model": data["model"],
        "track_id": data["track_id"],
        "frame_count": len(predictions),

        "fake_frame_count": fake_count,
        "real_frame_count": real_count,

        "fake_frame_percentage": (
            100.0
            * fake_count
            / len(predictions)
        ),

        "real_frame_percentage": (
            100.0
            * real_count
            / len(predictions)
        ),

        "fake_probability": {
            "mean": mean(fake_probs),
            "median": median(fake_probs),
            "p10": percentile(
                fake_probs,
                0.10,
            ),
            "p90": percentile(
                fake_probs,
                0.90,
            ),
            "minimum": min(fake_probs),
            "maximum": max(fake_probs),
        },

        "temporal": {
            "longest_fake_run_frames":
                longest_run(
                    labels,
                    "fake",
                ),

            "longest_real_run_frames":
                longest_run(
                    labels,
                    "real",
                ),

            "longest_fake_run_seconds":
                longest_run(
                    labels,
                    "fake",
                ) / 5.0,

            "longest_real_run_seconds":
                longest_run(
                    labels,
                    "real",
                ) / 5.0,
        },

        "first_frame": predictions[0],

        "last_frame": predictions[-1],

        "class_mapping": {
            "real": data["real_class"],
            "fake": data["fake_class"],
        },

        "frame_predictions_file":
            str(INPUT_PATH),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "Video-level analysis saved:"
    )
    print(OUTPUT_PATH)

    print()
    print(
        "Frames:",
        result["frame_count"],
    )

    print(
        "Fake frames:",
        result["fake_frame_count"],
    )

    print(
        "Real frames:",
        result["real_frame_count"],
    )

    print(
        "Fake percentage:",
        f"{result['fake_frame_percentage']:.2f}%",
    )

    print(
        "Mean fake probability:",
        f"{result['fake_probability']['mean']:.6f}",
    )

    print(
        "Median fake probability:",
        f"{result['fake_probability']['median']:.6f}",
    )

    print(
        "P10 fake probability:",
        f"{result['fake_probability']['p10']:.6f}",
    )

    print(
        "P90 fake probability:",
        f"{result['fake_probability']['p90']:.6f}",
    )

    print(
        "Longest fake run:",
        result["temporal"][
            "longest_fake_run_frames"
        ],
        "frames",
    )

    print(
        "Longest real run:",
        result["temporal"][
            "longest_real_run_frames"
        ],
        "frames",
    )


if __name__ == "__main__":
    main()
