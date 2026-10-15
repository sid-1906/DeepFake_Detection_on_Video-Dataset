from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image


# Use the exact GenD implementation downloaded
# from the official GenD checkpoint.
sys.path.insert(0, "model_store/gend")

from modeling_gend import GenD


MODEL_ID = "yermandy/GenD_CLIP_L_14"

IMAGE_PATH = Path(
    "data/processed/faces/track_0/frame_000000.jpg"
)


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Aligned face not found: {IMAGE_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print()

    print("Loading GenD:")
    print(MODEL_ID)
    print()

    model = GenD.from_pretrained(
        MODEL_ID
    )

    print("GenD checkpoint loaded.")

    model = model.to("cuda")
    model.eval()

    print("GenD moved to CUDA.")
    print()

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    tensor = (
        model.feature_extractor.preprocess(
            image
        )
    )

    tensor = tensor.unsqueeze(0).to("cuda")

    print("Input shape:", tuple(tensor.shape))
    print("Input dtype:", tensor.dtype)
    print("Input device:", tensor.device)

    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

    print()
    print("Logits:")
    print(logits.detach().cpu())

    print()
    print("Probabilities:")
    print(probabilities.detach().cpu())

    predicted_class = int(
        probabilities.argmax(dim=-1)[0]
    )

    print()
    print("Predicted class:", predicted_class)

    print()
    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated()
            / 1024**3,
            2,
        ),
        "GB",
    )

    print(
        "GPU memory reserved:",
        round(
            torch.cuda.memory_reserved()
            / 1024**3,
            2,
        ),
        "GB",
    )


if __name__ == "__main__":
    main()
