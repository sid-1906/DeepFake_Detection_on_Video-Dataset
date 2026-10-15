from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_ffprobe(video_path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration,size,format_name:"
            "stream=index,codec_name,codec_type,"
            "width,height,r_frame_rate,avg_frame_rate"
        ),
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def inspect_video(video_path: Path) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not video_path.is_file():
        raise ValueError(
            f"Path is not a regular file: {video_path}"
        )

    metadata = run_ffprobe(video_path)

    streams = metadata.get("streams", [])
    format_info = metadata.get("format", {})

    video_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "video"
    ]

    audio_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "audio"
    ]

    if not video_streams:
        raise ValueError("No video stream found.")

    video = video_streams[0]

    return {
        "valid": True,
        "file": video_path.name,
        "format": format_info.get("format_name"),
        "size_bytes": int(format_info["size"]),
        "duration_seconds": float(
            format_info["duration"]
        ),
        "video": {
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "frame_rate": video.get("avg_frame_rate"),
        },
        "audio": {
            "present": bool(audio_streams),
            "codec": (
                audio_streams[0].get("codec_name")
                if audio_streams
                else None
            ),
        },
    }


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/inspect_video.py <video>"
        )
        raise SystemExit(1)

    video_path = Path(sys.argv[1]).resolve()

    try:
        result = inspect_video(video_path)
    except (
        FileNotFoundError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(
            json.dumps(
                {
                    "valid": False,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        raise SystemExit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
