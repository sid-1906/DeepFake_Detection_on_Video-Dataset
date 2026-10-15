#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from multimodal_dataset import FFPPMultimodalDataset
from multimodal_model import MultimodalDeepfakeDetector


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "metadata"
    / "face_samples_train.json"
)

VAL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "metadata"
    / "face_samples_val.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "multimodal_best.pt"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "multimodal_final.pt"
)


IMAGE_SIZE = 224
SEQUENCE_LENGTH = 8

BATCH_SIZE = 4

EPOCHS = 15

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4

FEATURE_DIM = 256

NUM_WORKERS = 0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("Device:", device)
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        return device

    device = torch.device("cpu")

    print("Device:", device)

    return device


def get_labels(dataset):

    labels = []

    for record in dataset.records:

        label = record["label"].lower()

        if label == "real":
            labels.append(0)

        elif label == "fake":
            labels.append(1)

        else:
            raise ValueError(
                f"Unknown label: {record['label']}"
            )

    return torch.tensor(
        labels,
        dtype=torch.long,
    )


def create_balanced_sampler(dataset):

    labels = get_labels(dataset)

    class_counts = torch.bincount(
        labels,
        minlength=2,
    ).float()

    if torch.any(class_counts == 0):

        raise RuntimeError(
            "Training dataset is missing one of the classes."
        )

    print()
    print("Training class distribution")
    print("-" * 70)

    print(
        f"Real videos: {int(class_counts[0].item())}"
    )

    print(
        f"Fake videos: {int(class_counts[1].item())}"
    )

    # Inverse-frequency class weights.
    #
    # This makes REAL and FAKE approximately equally
    # represented during training.
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    print()
    print("Balanced sampler")
    print("-" * 70)

    print(
        "REAL sampling weight:",
        f"{class_weights[0].item():.6f}"
    )

    print(
        "FAKE sampling weight:",
        f"{class_weights[1].item():.6f}"
    )

    return sampler


def calculate_metrics(
    true_negative,
    false_positive,
    false_negative,
    true_positive,
):

    total = (
        true_negative
        + false_positive
        + false_negative
        + true_positive
    )

    correct = (
        true_negative
        + true_positive
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # REAL = class 0
    # --------------------------------------------------------

    real_precision_denominator = (
        true_negative
        + false_negative
    )

    real_recall_denominator = (
        true_negative
        + false_positive
    )

    real_precision = (
        true_negative
        / real_precision_denominator
        if real_precision_denominator > 0
        else 0.0
    )

    real_recall = (
        true_negative
        / real_recall_denominator
        if real_recall_denominator > 0
        else 0.0
    )

    real_f1_denominator = (
        real_precision
        + real_recall
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
    # FAKE = class 1
    # --------------------------------------------------------

    fake_precision_denominator = (
        true_positive
        + false_positive
    )

    fake_recall_denominator = (
        true_positive
        + false_negative
    )

    fake_precision = (
        true_positive
        / fake_precision_denominator
        if fake_precision_denominator > 0
        else 0.0
    )

    fake_recall = (
        true_positive
        / fake_recall_denominator
        if fake_recall_denominator > 0
        else 0.0
    )

    fake_f1_denominator = (
        fake_precision
        + fake_recall
    )

    fake_f1 = (
        2.0
        * fake_precision
        * fake_recall
        / fake_f1_denominator
        if fake_f1_denominator > 0
        else 0.0
    )

    macro_f1 = (
        real_f1 + fake_f1
    ) / 2.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,

        "real_precision": real_precision,
        "real_recall": real_recall,
        "real_f1": real_f1,

        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,

        "macro_f1": macro_f1,
    }


def print_metrics(
    loss,
    metrics,
):

    print()
    print(
        f"Val loss:       {loss:.4f}"
    )

    print(
        f"Val accuracy:   "
        f"{metrics['accuracy']:.4f}"
    )

    print()

    print(
        f"REAL precision: "
        f"{metrics['real_precision']:.4f}"
    )

    print(
        f"REAL recall:    "
        f"{metrics['real_recall']:.4f}"
    )

    print(
        f"REAL F1:        "
        f"{metrics['real_f1']:.4f}"
    )

    print()

    print(
        f"FAKE precision: "
        f"{metrics['fake_precision']:.4f}"
    )

    print(
        f"FAKE recall:    "
        f"{metrics['fake_recall']:.4f}"
    )

    print(
        f"FAKE F1:        "
        f"{metrics['fake_f1']:.4f}"
    )

    print()

    print(
        f"Macro F1:       "
        f"{metrics['macro_f1']:.4f}"
    )


def print_confusion_matrix(
    true_negative,
    false_positive,
    false_negative,
    true_positive,
):

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
        f"{true_negative:5d}   "
        f"{false_positive:5d}"
    )

    print(
        f"Actual FAKE      "
        f"{false_negative:5d}   "
        f"{true_positive:5d}"
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs,
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    print()
    print(
        f"Epoch {epoch}/{total_epochs}"
    )
    print("=" * 70)

    print()
    print("Training")
    print("-" * 70)

    for step, batch in enumerate(
        loader,
        start=1,
    ):

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

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            rgb,
            frequency,
            motion,
        )

        loss = criterion(
            logits,
            labels,
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        # Prevent occasional unstable gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += loss.item()

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if step % 50 == 0:

            print(
                f"Step {step:05d} | "
                f"Loss {loss.item():.4f}"
            )

    epoch_loss = (
        running_loss / len(loader)
    )

    epoch_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    print()
    print(
        f"Train loss:     "
        f"{epoch_loss:.4f}"
    )

    print(
        f"Train accuracy: "
        f"{epoch_accuracy:.4f}"
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    print()
    print("Validation")
    print("-" * 70)

    with torch.no_grad():

        for step, batch in enumerate(
            loader,
            start=1,
        ):

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

            logits = model(
                rgb,
                frequency,
                motion,
            )

            loss = criterion(
                logits,
                labels,
            )

            running_loss += loss.item()

            predictions = logits.argmax(
                dim=1
            )

            # ------------------------------------------------
            # Confusion matrix
            #
            # 0 = REAL
            # 1 = FAKE
            # ------------------------------------------------

            true_negative += (
                ((predictions == 0) &
                 (labels == 0))
                .sum()
                .item()
            )

            false_positive += (
                ((predictions == 1) &
                 (labels == 0))
                .sum()
                .item()
            )

            false_negative += (
                ((predictions == 0) &
                 (labels == 1))
                .sum()
                .item()
            )

            true_positive += (
                ((predictions == 1) &
                 (labels == 1))
                .sum()
                .item()
            )

            if step % 50 == 0:

                print(
                    f"Evaluated batches: "
                    f"{step}"
                )

    val_loss = (
        running_loss / len(loader)
    )

    metrics = calculate_metrics(
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )

    print_metrics(
        val_loss,
        metrics,
    )

    print_confusion_matrix(
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )

    return (
        val_loss,
        metrics,
        (
            true_negative,
            false_positive,
            false_negative,
            true_positive,
        ),
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_macro_f1,
):

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "feature_dim":
                FEATURE_DIM,

            "sequence_length":
                SEQUENCE_LENGTH,

            "image_size":
                IMAGE_SIZE,

            "epoch":
                epoch,

            "best_macro_f1":
                best_macro_f1,

            "pretrained_rgb":
                True,
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MULTIMODAL DEEPFAKE TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = FFPPMultimodalDataset(
        TRAIN_MANIFEST,
        image_size=IMAGE_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )

    val_dataset = FFPPMultimodalDataset(
        VAL_MANIFEST,
        image_size=IMAGE_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )

    print()
    print(
        "Training videos:",
        len(train_dataset),
    )

    print(
        "Validation videos:",
        len(val_dataset),
    )

    print(
        "Training records skipped:",
        train_dataset.skipped_records,
    )

    print(
        "Validation records skipped:",
        val_dataset.skipped_records,
    )

    # --------------------------------------------------------
    # Balanced training sampler
    # --------------------------------------------------------

    train_sampler = create_balanced_sampler(
        train_dataset
    )

    # --------------------------------------------------------
    # Training loader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # --------------------------------------------------------
    # Validation loader
    #
    # IMPORTANT:
    # Validation remains in its natural distribution.
    # We do NOT use the balanced sampler here.
    # --------------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating multimodal model...")
    print(
        "RGB backbone: ImageNet pretrained"
    )

    model = MultimodalDeepfakeDetector(
        feature_dim=FEATURE_DIM,
        pretrained_rgb=True,
    ).to(device)

    # --------------------------------------------------------
    # Loss
    #
    # Balanced sampler already handles the class imbalance.
    # Do NOT apply another class weighting here.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    #
    # Cosine decay gives the pretrained backbone a gentle
    # learning-rate reduction over the training run.
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Remove stale "best" model
    #
    # The previous multimodal_best.pt was the collapsed
    # FAKE-only model, so it must not be treated as a
    # starting point for this corrected training run.
    # --------------------------------------------------------

    if BEST_CHECKPOINT.exists():

        print()
        print(
            "Removing previous best checkpoint:"
        )

        print(
            BEST_CHECKPOINT
        )

        BEST_CHECKPOINT.unlink()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                total_epochs=EPOCHS,
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            metrics,
            confusion,
        ) = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        macro_f1 = metrics["macro_f1"]

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": metrics["accuracy"],
                "real_f1": metrics["real_f1"],
                "fake_f1": metrics["fake_f1"],
                "macro_f1": macro_f1,
            }
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1
            best_epoch = epoch

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_macro_f1=best_macro_f1,
            )

            print()
            print(
                "*** NEW BEST CHECKPOINT ***"
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            print(
                f"Best validation Macro F1: "
                f"{best_macro_f1:.4f}"
            )

            print(
                "Saved:",
                BEST_CHECKPOINT,
            )

        # ----------------------------------------------------
        # Learning-rate update
        # ----------------------------------------------------

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print()
        print(
            f"Learning rate: {current_lr:.8f}"
        )

    # --------------------------------------------------------
    # Save final epoch
    # --------------------------------------------------------

    save_checkpoint(
        path=FINAL_CHECKPOINT,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=EPOCHS,
        best_macro_f1=best_macro_f1,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch:        {best_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{best_macro_f1:.4f}"
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print()
    print(
        "Final checkpoint:"
    )

    print(
        FINAL_CHECKPOINT
    )

    print()
    print("=" * 70)
    print("EPOCH SUMMARY")
    print("=" * 70)

    for record in history:

        print(
            f"Epoch {record['epoch']:02d} | "
            f"Train Loss {record['train_loss']:.4f} | "
            f"Train Acc {record['train_accuracy']:.4f} | "
            f"Val Loss {record['val_loss']:.4f} | "
            f"Val Acc {record['val_accuracy']:.4f} | "
            f"REAL F1 {record['real_f1']:.4f} | "
            f"FAKE F1 {record['fake_f1']:.4f} | "
            f"Macro F1 {record['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()