from __future__ import annotations

import json
from pathlib import Path


INPUT_PATH = Path("data/processed/detections.json")
OUTPUT_PATH = Path("data/processed/tracks.json")

IOU_THRESHOLD = 0.30
MAX_MISSED_FRAMES = 2


def iou(box_a: dict, box_b: dict) -> float:
    ax1 = box_a["x"]
    ay1 = box_a["y"]
    ax2 = ax1 + box_a["width"]
    ay2 = ay1 + box_a["height"]

    bx1 = box_b["x"]
    by1 = box_b["y"]
    bx2 = bx1 + box_b["width"]
    by2 = by1 + box_b["height"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def create_track(track_id: int, detection: dict, frame: dict) -> dict:
    return {
        "track_id": track_id,
        "first_frame": frame["index"],
        "last_frame": frame["index"],
        "last_timestamp_seconds": frame["timestamp_seconds"],
        "missed_frames": 0,
        "detections": [
            {
                "frame_index": frame["index"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "file": frame["file"],
                "detection": detection,
            }
        ],
    }


def main() -> None:
    data = json.loads(INPUT_PATH.read_text())

    active_tracks: list[dict] = []
    finished_tracks: list[dict] = []

    next_track_id = 0

    for frame in data["frames"]:
        detections = frame["faces"]

        matched_track_ids: set[int] = set()
        matched_detection_ids: set[int] = set()

        candidates = []

        for track_index, track in enumerate(active_tracks):
            if not track["detections"]:
                continue

            previous_detection = track["detections"][-1]["detection"]

            for detection_index, detection in enumerate(detections):
                score = iou(
                    previous_detection["bbox"],
                    detection["bbox"],
                )

                if score >= IOU_THRESHOLD:
                    candidates.append(
                        (
                            score,
                            track_index,
                            detection_index,
                        )
                    )

        candidates.sort(
            reverse=True,
            key=lambda item: item[0],
        )

        for (
            score,
            track_index,
            detection_index,
        ) in candidates:

            track = active_tracks[track_index]

            if track["track_id"] in matched_track_ids:
                continue

            if detection_index in matched_detection_ids:
                continue

            detection = detections[detection_index]

            track["detections"].append(
                {
                    "frame_index": frame["index"],
                    "timestamp_seconds": (
                        frame["timestamp_seconds"]
                    ),
                    "file": frame["file"],
                    "detection": detection,
                }
            )

            track["last_frame"] = frame["index"]
            track["last_timestamp_seconds"] = (
                frame["timestamp_seconds"]
            )
            track["missed_frames"] = 0

            matched_track_ids.add(track["track_id"])
            matched_detection_ids.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_ids:
                continue

            track = create_track(
                next_track_id,
                detection,
                frame,
            )

            active_tracks.append(track)

            matched_track_ids.add(next_track_id)

            next_track_id += 1

        still_active = []

        for track in active_tracks:
            if track["track_id"] in matched_track_ids:
                still_active.append(track)
                continue

            track["missed_frames"] += 1

            if track["missed_frames"] > MAX_MISSED_FRAMES:
                finished_tracks.append(track)
            else:
                still_active.append(track)

        active_tracks = still_active

    finished_tracks.extend(active_tracks)

    finished_tracks.sort(
        key=lambda track: track["track_id"]
    )

    output = {
        "source": str(INPUT_PATH),
        "track_count": len(finished_tracks),
        "tracks": finished_tracks,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2)
    )

    print(
        f"Created {len(finished_tracks)} tracks."
    )

    for track in finished_tracks:
        print(
            f"Track {track['track_id']}: "
            f"{len(track['detections'])} detections, "
            f"frames "
            f"{track['first_frame']}-"
            f"{track['last_frame']}"
        )


if __name__ == "__main__":
    main()
