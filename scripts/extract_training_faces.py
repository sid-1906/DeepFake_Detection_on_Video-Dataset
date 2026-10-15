#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "splits"
    / "ffpp_splits.csv"
)

YU_NET = (
    PROJECT_ROOT
    / "model_store"
    / "face_detection"
    / "face_detection_yunet_2023mar.onnx"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "processed"
    / "faces"
)

SEED = 42
FRAMES_PER_VIDEO = 8

# Face crop size used by the eventual classifier.
CROP_SIZE = 224

# Minimum YuNet confidence.
CONFIDENCE_THRESHOLD = 0.6


def create_detector():
    detector = cv2.FaceDetectorYN.create(
        str(YU_NET),
        "",
        (320, 320),
        CONFIDENCE_THRESHOLD,
        0.3,
        5000,
    )
    return detector


def sample_frame_indices(frame_count, number):
    if frame_count <= 0:
        return []

    if frame_count <= number:
        return list(range(frame_count))

    # Evenly distributed samples.
    indices = []

    for i in range(number):
        position = (i + 0.5) / number
        index = int(position * frame_count)

        index = min(index, frame_count - 1)

        indices.append(index)

    return sorted(set(indices))


def crop_primary_face(frame, detector):
    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None, None

    # YuNet format:
    # x, y, w, h, landmarks..., confidence
    #
    # Choose the largest detected face.
    best = max(
        faces,
        key=lambda face: float(face[2] * face[3]),
    )

    x, y, w, h = best[:4]

    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)

    if w <= 0 or h <= 0:
        return None, None

    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 <= x or y2 <= y:
        return None, None

    # Expand the bounding box slightly to preserve
    # useful facial context.
    cx = (x + x2) / 2.0
    cy = (y + y2) / 2.0

    side = max(x2 - x, y2 - y) * 1.25

    left = int(cx - side / 2)
    top = int(cy - side / 2)
    right = int(cx + side / 2)
    bottom = int(cy + side / 2)

    # Pad when crop extends outside image.
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)

    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )

        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top

    crop = frame[top:bottom, left:right]

    if crop.size == 0:
        return None, None

    crop = cv2.resize(
        crop,
        (CROP_SIZE, CROP_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    confidence = float(best[-1])

    return crop, confidence


def process_video(row, detector, output_root, frames_per_video):
    video_path = PROJECT_ROOT / row["path"]

    label = row["label"]
    method = row["method"]
    split = row["split"]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "path": row["path"],
            "split": split,
            "label": label,
            "method": method,
            "status": "video_open_failed",
            "frames_saved": 0,
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = sample_frame_indices(
        frame_count,
        frames_per_video,
    )

    video_id = Path(row["path"]).stem

    video_output = (
        output_root
        / split
        / label
        / method
        / video_id
    )

    video_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = 0
    detections_failed = 0

    records = []

    for frame_index in indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        ok, frame = cap.read()

        if not ok:
            continue

        crop, confidence = crop_primary_face(
            frame,
            detector,
        )

        if crop is None:
            detections_failed += 1
            continue

        filename = f"frame_{frame_index:06d}.jpg"

        output_path = video_output / filename

        ok = cv2.imwrite(
            str(output_path),
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if not ok:
            continue

        timestamp = frame_index / max(
            float(row.get("frame_rate", 25.0)),
            1.0,
        )

        records.append({
            "image": str(
                output_path.relative_to(PROJECT_ROOT)
            ),
            "frame_index": frame_index,
            "timestamp_seconds": timestamp,
            "face_confidence": confidence,
        })

        saved += 1

    cap.release()

    return {
        "path": row["path"],
        "split": split,
        "label": label,
        "method": method,
        "source_group": row["source_group"],
        "status": "ok",
        "requested_frames": len(indices),
        "frames_saved": saved,
        "face_detection_failures": detections_failed,
        "frames": records,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N videos.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=FRAMES_PER_VIDEO,
        help="Number of evenly distributed frames sampled from each video.",
    )

    args = parser.parse_args()
    frames_per_video = args.frames_per_video

    random.seed(args.seed)

    if args.frames_per_video <= 0:
        raise ValueError("--frames-per-video must be greater than 0")

    df = pd.read_csv(MANIFEST)

    df = df[df["split"] == args.split].copy()

    # Randomize video order reproducibly.
    df = df.sample(
        frac=1.0,
        random_state=args.seed,
    ).reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit)

    print("FF++ face extraction")
    print("=" * 70)
    print(f"Split:             {args.split}")
    print(f"Videos:            {len(df)}")
    print(f"Frames/video:      {FRAMES_PER_VIDEO}")
    print(f"Face crop size:    {CROP_SIZE}x{CROP_SIZE}")
    print(f"YuNet threshold:   {CONFIDENCE_THRESHOLD}")
    print()

    detector = create_detector()

    output_root = OUTPUT_ROOT

    all_results = []

    for number, (_, row) in enumerate(df.iterrows(), start=1):

        result = process_video(
            row,
            detector,
            output_root,
            frames_per_video,
        )

        all_results.append(result)

        print(
            f"[{number:04d}/{len(df):04d}] "
            f"{row['label']:4s} "
            f"{row['method']:17s} "
            f"saved={result['frames_saved']}/{result.get('requested_frames', 0)} "
            f"{Path(row['path']).name}"
        )

    manifest_dir = (
        PROJECT_ROOT
        / "data"
        / "dataset"
        / "metadata"
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_manifest = (
        manifest_dir
        / f"face_samples_{args.split}.json"
    )

    with open(output_manifest, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    total_requested = sum(
        r.get("requested_frames", 0)
        for r in all_results
    )

    total_saved = sum(
        r.get("frames_saved", 0)
        for r in all_results
    )

    total_failed = sum(
        r.get("face_detection_failures", 0)
        for r in all_results
    )

    print()
    print("=" * 70)
    print("Extraction complete")
    print("=" * 70)
    print(f"Videos processed:        {len(all_results)}")
    print(f"Frames requested:        {total_requested}")
    print(f"Face crops saved:        {total_saved}")
    print(f"Face detections failed:  {total_failed}")
    print(f"Manifest:                {output_manifest}")


if __name__ == "__main__":
    main()
