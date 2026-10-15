from __future__ import annotations

import json
import sys
from pathlib import Path

from inspect_video import inspect_video


MAX_FILE_SIZE_MB = 500
MAX_DURATION_SECONDS = 300
MIN_WIDTH = 320
MIN_HEIGHT = 240
MAX_WIDTH = 7680
MAX_HEIGHT = 4320

SUPPORTED_VIDEO_CODECS = {
    "h264",
    "hevc",
    "vp8",
    "vp9",
    "av1",
}


def validate(metadata: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if metadata["size_bytes"] > MAX_FILE_SIZE_MB * 1024 * 1024:
        errors.append(
            f"File exceeds {MAX_FILE_SIZE_MB} MB limit."
        )

    duration = metadata["duration_seconds"]

    if duration <= 0:
        errors.append("Video duration must be greater than zero.")

    if duration > MAX_DURATION_SECONDS:
        errors.append(
            f"Video exceeds {MAX_DURATION_SECONDS} second limit."
        )

    video = metadata["video"]

    width = video["width"]
    height = video["height"]
    codec = video["codec"]

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        errors.append(
            f"Resolution is below "
            f"{MIN_WIDTH}x{MIN_HEIGHT}."
        )

    if width > MAX_WIDTH or height > MAX_HEIGHT:
        errors.append(
            f"Resolution exceeds "
            f"{MAX_WIDTH}x{MAX_HEIGHT}."
        )

    if codec not in SUPPORTED_VIDEO_CODECS:
        errors.append(
            f"Unsupported video codec: {codec}"
        )

    return len(errors) == 0, errors


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/validate_video.py <video>"
        )
        raise SystemExit(1)

    video_path = Path(sys.argv[1]).resolve()

    try:
        metadata = inspect_video(video_path)
        valid, errors = validate(metadata)

        result = {
            "valid": valid,
            "errors": errors,
            "metadata": metadata,
        }

        print(json.dumps(result, indent=2))

        raise SystemExit(0 if valid else 1)

    except Exception as exc:
        print(
            json.dumps(
                {
                    "valid": False,
                    "errors": [str(exc)],
                },
                indent=2,
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
