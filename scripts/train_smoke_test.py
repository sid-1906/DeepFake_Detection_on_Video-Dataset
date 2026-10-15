#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from multimodal_dataset import FFPPMultimodalDataset
from multimodal_model import MultimodalDeepfakeDetector


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("MULTIMODAL MODEL SMOKE TEST")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    dataset = FFPPMultimodalDataset(
        "data/dataset/metadata/face_samples_train.json",
        image_size=224,
        sequence_length=8,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultimodalDeepfakeDetector(
        feature_dim=256,
        pretrained_rgb=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    model.train()

    batch = next(iter(loader))

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

    print()
    print("Input tensors")
    print("-" * 70)
    print("RGB:", tuple(rgb.shape))
    print("Frequency:", tuple(frequency.shape))
    print("Motion:", tuple(motion.shape))
    print("Labels:", tuple(labels.shape))

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

    print()
    print("Forward pass")
    print("-" * 70)
    print("Logits:", tuple(logits.shape))
    print("Loss:", float(loss))

    loss.backward()

    optimizer.step()

    print()
    print("Backward pass: PASS")
    print("Optimizer step: PASS")

    predictions = logits.argmax(
        dim=1
    )

    accuracy = (
        predictions == labels
    ).float().mean()

    print(
        "Batch accuracy:",
        float(accuracy),
    )

    if torch.cuda.is_available():
        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print()
        print("GPU memory")
        print("-" * 70)
        print(
            f"Allocated: {allocated:.2f} GB"
        )
        print(
            f"Reserved:  {reserved:.2f} GB"
        )

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
