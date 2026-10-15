#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts"),
)


from multimodal_dataset import (
    FFPPMultimodalDataset,
)

from multimodal_model import (
    MultimodalDeepfakeDetector,
)


# ============================================================
# FILES
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "multimodal_best.pt"
)

TEST_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "metadata"
    / "face_samples_test.json"
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    tn,
    fp,
    fn,
    tp,
):
    """
    Compute binary classification metrics.

    Label convention used by the dataset:
        0 = REAL
        1 = FAKE
    """

    total = tn + fp + fn + tp

    accuracy = (
        (tn + tp) / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # REAL CLASS
    #
    # REAL is treated as class 0.
    #
    # precision = TN / (TN + FN)
    # recall    = TN / (TN + FP)
    # --------------------------------------------------------

    real_precision_denominator = tn + fn

    real_recall_denominator = tn + fp

    real_precision = (
        tn / real_precision_denominator
        if real_precision_denominator > 0
        else 0.0
    )

    real_recall = (
        tn / real_recall_denominator
        if real_recall_denominator > 0
        else 0.0
    )

    real_f1_denominator = (
        real_precision + real_recall
    )

    real_f1 = (
        2.0
        * real_precision
        * real_recall
        / real_f1_denominator
        if real_f1_denominator > 0
        else 0.0
    )

    # --------------------------------------------------------
    # FAKE CLASS
    #
    # FAKE is class 1.
    #
    # precision = TP / (TP + FP)
    # recall    = TP / (TP + FN)
    # --------------------------------------------------------

    fake_precision_denominator = tp + fp

    fake_recall_denominator = tp + fn

    fake_precision = (
        tp / fake_precision_denominator
        if fake_precision_denominator > 0
        else 0.0
    )

    fake_recall = (
        tp / fake_recall_denominator
        if fake_recall_denominator > 0
        else 0.0
    )

    fake_f1_denominator = (
        fake_precision + fake_recall
    )

    fake_f1 = (
        2.0
        * fake_precision
        * fake_recall
        / fake_f1_denominator
        if fake_f1_denominator > 0
        else 0.0
    )

    # --------------------------------------------------------
    # MACRO F1
    # --------------------------------------------------------

    macro_f1 = (
        real_f1 + fake_f1
    ) / 2.0

    return {
        "accuracy": accuracy,

        "real_precision": real_precision,
        "real_recall": real_recall,
        "real_f1": real_f1,

        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,

        "macro_f1": macro_f1,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MULTIMODAL DEEPFAKE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    if not TEST_MANIFEST.exists():

        raise FileNotFoundError(
            f"Test manifest not found:\n{TEST_MANIFEST}"
        )

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    print()
    print("Test manifest:")
    print(TEST_MANIFEST)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = FFPPMultimodalDataset(
        TEST_MANIFEST,
        image_size=224,
        sequence_length=8,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print(
        "Test videos:",
        len(dataset),
    )

    print(
        "Records skipped:",
        getattr(
            dataset,
            "skipped_records",
            0,
        ),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # The checkpoint was created with:
    #
    #   feature_dim = 256
    #   sequence_length = 8
    #   pretrained_rgb = False
    #
    # Therefore evaluation must use the same architecture.
    #
    # --------------------------------------------------------

    model = MultimodalDeepfakeDetector(
        feature_dim=256,
        pretrained_rgb=False,
        sequence_length=8,
    ).to(device)

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # CHECKPOINT INFORMATION
    # --------------------------------------------------------

    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"],
        )

    if "feature_dim" in checkpoint:

        print(
            "Feature dimension:",
            checkpoint["feature_dim"],
        )

    if "sequence_length" in checkpoint:

        print(
            "Sequence length:",
            checkpoint["sequence_length"],
        )

    if "image_size" in checkpoint:

        print(
            "Image size:",
            checkpoint["image_size"],
        )

    if "pretrained_rgb" in checkpoint:

        print(
            "Pretrained RGB:",
            checkpoint["pretrained_rgb"],
        )

    if "best_macro_f1" in checkpoint:

        print(
            "Best validation Macro F1:",
            f"{checkpoint['best_macro_f1']:.4f}",
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATING TEST SET")
    print("=" * 70)

    total = 0
    correct = 0

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    with torch.no_grad():

        for step, batch in enumerate(
            loader,
            start=1,
        ):

            # ------------------------------------------------
            # INPUTS
            # ------------------------------------------------

            rgb = batch["rgb"].to(
                device,
                non_blocking=True,
            )

            frequency = batch["frequency"].to(
                device,
                non_blocking=True,
            )

            motion = batch["motion"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            logits = model(
                rgb,
                frequency,
                motion,
            )

            predictions = logits.argmax(
                dim=1
            )

            # ------------------------------------------------
            # OVERALL
            # ------------------------------------------------

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # ------------------------------------------------
            # CONFUSION MATRIX
            #
            # 0 = REAL
            # 1 = FAKE
            # ------------------------------------------------

            tn += (
                (
                    (predictions == 0)
                    & (labels == 0)
                )
                .sum()
                .item()
            )

            fp += (
                (
                    (predictions == 1)
                    & (labels == 0)
                )
                .sum()
                .item()
            )

            fn += (
                (
                    (predictions == 0)
                    & (labels == 1)
                )
                .sum()
                .item()
            )

            tp += (
                (
                    (predictions == 1)
                    & (labels == 1)
                )
                .sum()
                .item()
            )

            if step % 50 == 0:

                print(
                    f"Evaluated batches: {step}"
                )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = compute_metrics(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Samples:        {total}"
    )

    print(
        f"Correct:        {correct}"
    )

    print(
        f"Incorrect:      {total - correct}"
    )

    print(
        f"Accuracy:       "
        f"{metrics['accuracy']:.4f}"
    )

    print()
    print("REAL CLASS")
    print("-" * 70)

    print(
        f"Precision:      "
        f"{metrics['real_precision']:.4f}"
    )

    print(
        f"Recall:         "
        f"{metrics['real_recall']:.4f}"
    )

    print(
        f"F1:             "
        f"{metrics['real_f1']:.4f}"
    )

    print()
    print("FAKE CLASS")
    print("-" * 70)

    print(
        f"Precision:      "
        f"{metrics['fake_precision']:.4f}"
    )

    print(
        f"Recall:         "
        f"{metrics['fake_recall']:.4f}"
    )

    print(
        f"F1:             "
        f"{metrics['fake_f1']:.4f}"
    )

    print()
    print(
        f"Macro F1:       "
        f"{metrics['macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print()
    print("Confusion matrix")
    print("-" * 70)

    print(
        "                 Predicted"
    )

    print(
        "                 REAL    FAKE"
    )

    print(
        f"Actual REAL      "
        f"{tn:5d}   "
        f"{fp:5d}"
    )

    print(
        f"Actual FAKE      "
        f"{fn:5d}   "
        f"{tp:5d}"
    )

    # --------------------------------------------------------
    # CLASS COUNTS
    # --------------------------------------------------------

    print()
    print("Test-set class counts")
    print("-" * 70)

    print(
        f"Actual REAL:     {tn + fp}"
    )

    print(
        f"Actual FAKE:     {fn + tp}"
    )

    print(
        f"Predicted REAL:  {tn + fn}"
    )

    print(
        f"Predicted FAKE:  {fp + tp}"
    )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()