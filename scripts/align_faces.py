from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


TRACKS_PATH = Path("data/processed/tracks.json")
OUTPUT_ROOT = Path("data/processed/faces")

FACE_SIZE = 256

# Five-point canonical landmarks for a 256x256 aligned face.
REFERENCE_POINTS = np.array(
    [
        [82.0, 105.0],   # left eye
        [174.0, 105.0],  # right eye
        [128.0, 145.0],  # nose
        [92.0, 190.0],   # left mouth
        [164.0, 190.0],  # right mouth
    ],
    dtype=np.float32,
)


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    transform, _ = cv2.estimateAffinePartial2D(
        landmarks,
        REFERENCE_POINTS,
        method=cv2.LMEDS,
    )

    if transform is None:
        raise RuntimeError(
            "Could not estimate face alignment transform."
        )

    aligned = cv2.warpAffine(
        image,
        transform,
        (FACE_SIZE, FACE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aligned


def main() -> None:
    data = json.loads(TRACKS_PATH.read_text())

    if not data["tracks"]:
        raise RuntimeError("No tracks found.")

    # For this first pipeline pass, select the track with
    # the greatest number of detections.
    primary_track = max(
        data["tracks"],
        key=lambda track: len(track["detections"]),
    )

    track_id = primary_track["track_id"]

    output_dir = OUTPUT_ROOT / f"track_{track_id}"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for item in primary_track["detections"]:
        image_path = Path(item["file"])

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        landmarks = np.array(
            [
                [point["x"], point["y"]]
                for point in item["detection"]["landmarks"]
            ],
            dtype=np.float32,
        )

        if landmarks.shape != (5, 2):
            raise RuntimeError(
                f"Unexpected landmark shape: "
                f"{landmarks.shape}"
            )

        aligned = align_face(
            image,
            landmarks,
        )

        output_path = (
            output_dir
            / f"frame_{item['frame_index']:06d}.jpg"
        )

        success = cv2.imwrite(
            str(output_path),
            aligned,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not success:
            raise RuntimeError(
                f"Could not write: {output_path}"
            )

        records.append(
            {
                "frame_index": item["frame_index"],
                "timestamp_seconds": item[
                    "timestamp_seconds"
                ],
                "source_file": item["file"],
                "aligned_file": str(output_path),
                "confidence": item[
                    "detection"
                ]["confidence"],
            }
        )

    manifest = {
        "track_id": track_id,
        "face_size": FACE_SIZE,
        "frame_count": len(records),
        "frames": records,
    }

    manifest_path = output_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print(
        f"Aligned {len(records)} faces."
    )
    print(
        f"Primary track: {track_id}"
    )
    print(
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
