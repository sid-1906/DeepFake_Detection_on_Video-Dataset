#!/usr/bin/env python3

import argparse
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

INPUT_DIR = PROJECT_ROOT / "data" / "input"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run_command(command: list[str]) -> None:
    print()
    print("=" * 70)
    print("RUNNING:", " ".join(command))
    print("=" * 70)

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def clean_processed_data() -> None:
    targets = [
        PROCESSED_DIR / "frames",
        PROCESSED_DIR / "faces",
        PROCESSED_DIR / "detections.json",
        PROCESSED_DIR / "tracks.json",
        PROCESSED_DIR / "gend_predictions.json",
        PROCESSED_DIR / "video_result.json",
        PROCESSED_DIR / "final_result.json",
    ]

    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete SID video analysis pipeline."
    )

    parser.add_argument(
        "video",
        help="Input video filename inside data/input/",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frame sampling FPS. Default: 5.",
    )

    args = parser.parse_args()

    input_video = INPUT_DIR / args.video

    if not input_video.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_video}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("SID pipeline")
    print("-" * 70)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Input:   {input_video}")
    print(f"FPS:     {args.fps}")

    print()
    print("Cleaning previous processed outputs...")
    clean_processed_data()

    # 1. Validate input video
    run_command([
        "python",
        str(SCRIPTS_DIR / "validate_video.py"),
        str(input_video),
    ])

    # 2. Extract frames
    run_command([
        "python",
        str(SCRIPTS_DIR / "extract_frames.py"),
        str(input_video),
        "--output",
        str(PROCESSED_DIR / "frames"),
        "--fps",
        str(args.fps),
    ])

    # 3. Detect faces
    run_command([
        "python",
        str(SCRIPTS_DIR / "detect_faces.py"),
    ])

    # 4. Track faces
    run_command([
        "python",
        str(SCRIPTS_DIR / "track_faces.py"),
    ])

    # 5. Align primary face track
    run_command([
        "python",
        str(SCRIPTS_DIR / "align_faces.py"),
    ])

    # 6. Run GenD
    run_command([
        "python",
        str(SCRIPTS_DIR / "run_gend.py"),
    ])

    # 7. Analyze frame-level predictions
    run_command([
        "python",
        str(SCRIPTS_DIR / "analyze_gend.py"),
    ])

    # 8. Aggregate to video level
    run_command([
        "python",
        str(SCRIPTS_DIR / "aggregate_gend.py"),
    ])

    # 9. Build final API-style result
    run_command([
        "python",
        str(SCRIPTS_DIR / "build_final_result.py"),
    ])

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print(f"Final result: {PROCESSED_DIR / 'final_result.json'}")


if __name__ == "__main__":
    main()