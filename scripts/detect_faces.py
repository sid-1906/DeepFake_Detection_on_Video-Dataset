from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


MODEL_PATH = Path(
    "model_store/face_detection/"
    "face_detection_yunet_2023mar.onnx"
)


def detect_faces(
    image_path: Path,
    detector: cv2.FaceDetectorYN,
) -> list[dict]:
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        return []

    results = []

    for face_index, face in enumerate(faces):
        x, y, w, h = face[:4]

        landmarks = face[4:14].reshape(5, 2)

        results.append(
            {
                "face_index": face_index,
                "bbox": {
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                },
                "landmarks": [
                    {
                        "x": float(point[0]),
                        "y": float(point[1]),
                    }
                    for point in landmarks
                ],
                "confidence": float(face[14]),
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames",
        default="data/processed/frames",
    )

    parser.add_argument(
        "--output",
        default="data/processed/detections.json",
    )

    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
    )

    args = parser.parse_args()

    frames_dir = Path(args.frames)
    output_path = Path(args.output)
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {model_path}"
        )

    manifest_path = frames_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frame manifest not found: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text()
    )

    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (320, 320),
        0.6,
        0.3,
        5000,
    )

    output_frames = []

    total_faces = 0

    for frame in manifest["frames"]:
        image_path = Path(frame["file"])

        faces = detect_faces(
            image_path,
            detector,
        )

        total_faces += len(faces)

        output_frames.append(
            {
                "index": frame["index"],
                "timestamp_seconds": (
                    frame["timestamp_seconds"]
                ),
                "file": frame["file"],
                "faces": faces,
            }
        )

    output = {
        "source_video": manifest["source_video"],
        "frame_count": len(output_frames),
        "total_detections": total_faces,
        "frames": output_frames,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2)
    )

    print(
        f"Processed {len(output_frames)} frames."
    )

    print(
        f"Total face detections: {total_faces}"
    )

    print(
        f"Detection manifest: {output_path}"
    )


if __name__ == "__main__":
    main()
