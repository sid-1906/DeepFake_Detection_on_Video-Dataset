from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import torch
from PIL import Image


sys.path.insert(0, "model_store/gend")

from modeling_gend import GenD


MODEL_ID = "yermandy/GenD_CLIP_L_14"

DATA_DIR = Path("data/calibration/gend")
DATA_DIR.mkdir(parents=True, exist_ok=True)


SAMPLES = {
    "known_fake": (
        "https://github.com/yermandy/deepfake-detection/"
        "blob/main/datasets/FF/DF/000_003/000.png?raw=true"
    ),
    "known_real": (
        "https://github.com/yermandy/deepfake-detection/"
        "blob/main/datasets/FF/real/000/000.png?raw=true"
    ),
}


def download(url: str, output: Path) -> None:
    if output.exists():
        return

    print(f"Downloading: {output}")

    urllib.request.urlretrieve(
        url,
        output,
    )


def predict(model: GenD, image: Image.Image) -> torch.Tensor:
    tensor = model.feature_extractor.preprocess(
        image
    )

    tensor = tensor.unsqueeze(0).to("cuda")

    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

    return probabilities[0].detach().cpu()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print()

    print("Loading GenD...")
    model = GenD.from_pretrained(
        MODEL_ID
    )

    model = model.to("cuda")
    model.eval()

    print("GenD loaded.")
    print()

    results = {}

    for name, url in SAMPLES.items():
        path = DATA_DIR / f"{name}.png"

        download(url, path)

        image = Image.open(path).convert("RGB")

        probabilities = predict(
            model,
            image,
        )

        results[name] = probabilities

        print(name)
        print("  image:", path)
        print(
            "  class 0:",
            f"{probabilities[0].item():.6f}",
        )
        print(
            "  class 1:",
            f"{probabilities[1].item():.6f}",
        )
        print(
            "  predicted:",
            int(probabilities.argmax()),
        )
        print()

    fake_class = int(
        results["known_fake"].argmax()
    )

    real_class = int(
        results["known_real"].argmax()
    )

    print("=" * 60)

    print("Calibration result")

    print(
        "Known FAKE -> class",
        fake_class,
    )

    print(
        "Known REAL -> class",
        real_class,
    )

    if fake_class == real_class:
        raise RuntimeError(
            "Calibration failed: both known samples "
            "mapped to the same class."
        )

    print()

    print(
        f"FAKE_CLASS={fake_class}"
    )

    print(
        f"REAL_CLASS={real_class}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
