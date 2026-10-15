#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import json
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "raw"
    / "extracted"
    / "FaceForensics++_C23"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "original",
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

records = []

for method in METHODS:
    csv_path = DATASET_ROOT / "csv" / f"{method}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        relative_path = str(row["File Path"])
        video_path = DATASET_ROOT / relative_path

        if not video_path.exists():
            print(f"WARNING: missing {video_path}")
            continue

        filename = Path(relative_path).stem

        if method == "original":
            # 000.mp4 -> source group 000
            source_group = filename

        elif method in {
            "Deepfakes",
            "Face2Face",
            "FaceShifter",
            "FaceSwap",
            "NeuralTextures",
        }:
            # 000_003.mp4 -> source group 000
            source_group = filename.split("_")[0]

        else:
            # DeepFakeDetection does not have the same
            # simple source-video naming scheme.
            # Keep each source video isolated.
            source_group = f"DFD_{filename}"

        label = str(row["Label"]).strip().lower()

        records.append({
            "path": str(video_path.relative_to(PROJECT_ROOT)),
            "method": method,
            "label": label,
            "source_group": source_group,
            "frame_count": int(row["Frame Count"]),
            "width": int(row["Width"]),
            "height": int(row["Height"]),
            "codec": str(row["Codec"]),
            "file_size_mb": float(row["File Size(MB)"]),
        })


df = pd.DataFrame(records)

# Remove accidental duplicates.
df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)

print()
print("FF++ dataset manifest")
print("=" * 70)
print(f"Videos: {len(df)}")
print()
print("Labels:")
print(df["label"].value_counts().to_string())
print()
print("Methods:")
print(df["method"].value_counts().to_string())
print()
print("Unique source groups:", df["source_group"].nunique())

output_csv = OUTPUT_DIR / "ffpp_manifest.csv"
df.to_csv(output_csv, index=False)

summary = {
    "dataset": "FaceForensics++ C23",
    "videos": int(len(df)),
    "real_videos": int((df["label"] == "real").sum()),
    "fake_videos": int((df["label"] == "fake").sum()),
    "methods": {
        method: int((df["method"] == method).sum())
        for method in METHODS
    },
    "unique_source_groups": int(df["source_group"].nunique()),
}

with open(OUTPUT_DIR / "ffpp_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print(f"Manifest: {output_csv}")
print(f"Summary:  {OUTPUT_DIR / 'ffpp_summary.json'}")
