from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def get_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def extract_one_frame(
    video_path: Path,
    output_path: Path,
    timestamp: float,
) -> bool:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    return result.returncode == 0 and output_path.exists()


def extract_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration(video_path)
    interval = 1.0 / sample_fps

    records: list[dict] = []

    index = 0

    while True:
        timestamp = index * interval

        if timestamp >= duration:
            break

        timestamp = round(timestamp, 6)

        output_path = output_dir / f"frame_{index:06d}.jpg"

        success = extract_one_frame(
            video_path,
            output_path,
            timestamp,
        )

        if success:
            records.append(
                {
                    "index": index,
                    "timestamp_seconds": timestamp,
                    "file": str(output_path),
                }
            )

        index += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract timestamp-aware video frames."
    )

    parser.add_argument("video")

    parser.add_argument(
        "--output",
        default="data/processed/frames",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    output_dir = Path(args.output).resolve()

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if args.fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    records = extract_frames(
        video_path,
        output_dir,
        args.fps,
    )

    manifest_path = output_dir / "manifest.json"

    manifest = {
        "source_video": str(video_path),
        "sample_fps": args.fps,
        "duration_seconds": get_duration(video_path),
        "frame_count": len(records),
        "frames": records,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print(f"Extracted {len(records)} frames.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()