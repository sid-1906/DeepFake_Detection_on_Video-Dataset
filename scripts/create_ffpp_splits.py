#!/usr/bin/env python3

from pathlib import Path
import json
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "metadata"
    / "ffpp_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "splits"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


df = pd.read_csv(MANIFEST)

random.seed(SEED)

# ------------------------------------------------------------------
# Split at source-group level.
# ------------------------------------------------------------------

groups = sorted(df["source_group"].unique())

random.shuffle(groups)

n_groups = len(groups)

n_train = int(n_groups * TRAIN_RATIO)
n_val = int(n_groups * VAL_RATIO)

train_groups = set(groups[:n_train])
val_groups = set(groups[n_train:n_train + n_val])
test_groups = set(groups[n_train + n_val:])


def assign_split(group):
    if group in train_groups:
        return "train"
    if group in val_groups:
        return "val"
    return "test"


df["split"] = df["source_group"].map(assign_split)


# ------------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------------

assert set(df["split"]) == {"train", "val", "test"}

train_groups_check = set(
    df.loc[df["split"] == "train", "source_group"]
)

val_groups_check = set(
    df.loc[df["split"] == "val", "source_group"]
)

test_groups_check = set(
    df.loc[df["split"] == "test", "source_group"]
)

assert train_groups_check.isdisjoint(val_groups_check)
assert train_groups_check.isdisjoint(test_groups_check)
assert val_groups_check.isdisjoint(test_groups_check)


# ------------------------------------------------------------------
# Save individual split files.
# ------------------------------------------------------------------

for split in ["train", "val", "test"]:
    split_df = df[df["split"] == split].copy()

    output = OUTPUT_DIR / f"{split}.csv"
    split_df.to_csv(output, index=False)

    print(
        f"{split:5s}: "
        f"{len(split_df):4d} videos | "
        f"groups={split_df['source_group'].nunique():4d} | "
        f"real={(split_df['label'] == 'real').sum():4d} | "
        f"fake={(split_df['label'] == 'fake').sum():4d}"
    )


# ------------------------------------------------------------------
# Save combined manifest.
# ------------------------------------------------------------------

combined_output = OUTPUT_DIR / "ffpp_splits.csv"
df.to_csv(combined_output, index=False)


summary = {
    "dataset": "FaceForensics++ C23",
    "seed": SEED,
    "ratios": {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
    },
    "videos": {
        "total": len(df),
        "train": int((df["split"] == "train").sum()),
        "val": int((df["split"] == "val").sum()),
        "test": int((df["split"] == "test").sum()),
    },
    "source_groups": {
        "total": int(df["source_group"].nunique()),
        "train": len(train_groups_check),
        "val": len(val_groups_check),
        "test": len(test_groups_check),
    },
    "labels": {},
}

for split in ["train", "val", "test"]:
    subset = df[df["split"] == split]

    summary["labels"][split] = {
        "real": int((subset["label"] == "real").sum()),
        "fake": int((subset["label"] == "fake").sum()),
    }


with open(OUTPUT_DIR / "split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)


print()
print("Split verification")
print("=" * 70)
print("Train/validation overlap:",
      len(train_groups_check & val_groups_check))
print("Train/test overlap:",
      len(train_groups_check & test_groups_check))
print("Validation/test overlap:",
      len(val_groups_check & test_groups_check))

print()
print(f"Combined manifest: {combined_output}")
print(f"Summary:           {OUTPUT_DIR / 'split_summary.json'}")
