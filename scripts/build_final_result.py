from __future__ import annotations

import json
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/video_result.json"
)

OUTPUT_PATH = Path(
    "data/processed/final_result.json"
)


def main():
    data = json.loads(
        INPUT_PATH.read_text()
    )

    fake_percentage = (
        data["fake_frame_percentage"]
    )

    mean_fake_probability = (
        data["fake_probability"]["mean"]
    )

    median_fake_probability = (
        data["fake_probability"]["median"]
    )

    # This is a descriptive video-level label
    # based on the majority of sampled frame
    # classifications. It is NOT a calibrated
    # probability that the entire video is fake.
    video_label = (
        "fake"
        if data["fake_frame_count"]
        > data["real_frame_count"]
        else "real"
    )

    result = {
        "status": "completed",

        "model": {
            "name": data["model"],
            "track_id": data["track_id"],
        },

        "result": {
            "label": video_label,

            "sampled_frames": data[
                "frame_count"
            ],

            "fake_frames": data[
                "fake_frame_count"
            ],

            "real_frames": data[
                "real_frame_count"
            ],

            "fake_frame_percentage":
                fake_percentage,

            "real_frame_percentage":
                data[
                    "real_frame_percentage"
                ],
        },

        "evidence": {
            "mean_fake_probability":
                mean_fake_probability,

            "median_fake_probability":
                median_fake_probability,

            "p10_fake_probability":
                data[
                    "fake_probability"
                ]["p10"],

            "p90_fake_probability":
                data[
                    "fake_probability"
                ]["p90"],

            "minimum_fake_probability":
                data[
                    "fake_probability"
                ]["minimum"],

            "maximum_fake_probability":
                data[
                    "fake_probability"
                ]["maximum"],

            "longest_fake_run_frames":
                data[
                    "temporal"
                ]["longest_fake_run_frames"],

            "longest_fake_run_seconds":
                data[
                    "temporal"
                ]["longest_fake_run_seconds"],

            "longest_real_run_frames":
                data[
                    "temporal"
                ]["longest_real_run_frames"],

            "longest_real_run_seconds":
                data[
                    "temporal"
                ]["longest_real_run_seconds"],
        },

        "class_mapping": data[
            "class_mapping"
        ],

        "artifacts": {
            "frame_predictions":
                data[
                    "frame_predictions_file"
                ],

            "video_analysis":
                str(INPUT_PATH),
        },

        "note": (
            "The video label is a descriptive "
            "aggregation of sampled frame "
            "classifications. The probability "
            "statistics are GenD model outputs "
            "and should not be interpreted as "
            "a calibrated probability for the "
            "entire video."
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "Final result saved:"
    )
    print(OUTPUT_PATH)

    print()
    print(
        "Status:",
        result["status"],
    )

    print(
        "Video label:",
        result["result"]["label"],
    )

    print(
        "Sampled frames:",
        result["result"]["sampled_frames"],
    )

    print(
        "Fake frames:",
        result["result"]["fake_frames"],
    )

    print(
        "Real frames:",
        result["result"]["real_frames"],
    )

    print(
        "Mean fake probability:",
        f"{result['evidence']['mean_fake_probability']:.6f}",
    )


if __name__ == "__main__":
    main()
