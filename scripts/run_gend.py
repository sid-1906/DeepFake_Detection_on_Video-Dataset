from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image


sys.path.insert(0, "model_store/gend")

from modeling_gend import GenD


MODEL_ID = "yermandy/GenD_CLIP_L_14"

INPUT_MANIFEST = Path(
    "data/processed/faces/track_0/manifest.json"
)

OUTPUT_PATH = Path(
    "data/processed/gend_predictions.json"
)

FAKE_CLASS = 1
REAL_CLASS = 0


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing manifest: {INPUT_MANIFEST}"
        )

    data = json.loads(
        INPUT_MANIFEST.read_text()
    )

    print(
        "Loading GenD:",
        MODEL_ID,
    )

    model = GenD.from_pretrained(
        MODEL_ID
    )

    model = model.to("cuda")
    model.eval()

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Frames:",
        data["frame_count"],
    )

    predictions = []

    for i, frame in enumerate(data["frames"], start=1):
        image_path = Path(
            frame["aligned_file"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        tensor = (
            model.feature_extractor.preprocess(
                image
            )
        )

        tensor = tensor.unsqueeze(0).to(
            "cuda"
        )

        with torch.inference_mode():
            logits = model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0]

        real_probability = float(
            probabilities[REAL_CLASS]
            .detach()
            .cpu()
        )

        fake_probability = float(
            probabilities[FAKE_CLASS]
            .detach()
            .cpu()
        )

        predicted_class = int(
            probabilities.argmax()
            .detach()
            .cpu()
        )

        prediction = (
            "fake"
            if predicted_class == FAKE_CLASS
            else "real"
        )

        predictions.append(
            {
                "frame_index": frame[
                    "frame_index"
                ],
                "timestamp_seconds": frame[
                    "timestamp_seconds"
                ],
                "source_file": frame[
                    "source_file"
                ],
                "aligned_file": frame[
                    "aligned_file"
                ],
                "detection_confidence": frame[
                    "confidence"
                ],
                "real_probability": real_probability,
                "fake_probability": fake_probability,
                "predicted_class": predicted_class,
                "prediction": prediction,
            }
        )

        print(
            f"[{i:02d}/{data['frame_count']}] "
            f"t={frame['timestamp_seconds']:.1f}s "
            f"real={real_probability:.4f} "
            f"fake={fake_probability:.4f} "
            f"-> {prediction}"
        )

    output = {
        "model": MODEL_ID,
        "track_id": data["track_id"],
        "frame_count": len(predictions),
        "fake_class": FAKE_CLASS,
        "real_class": REAL_CLASS,
        "predictions": predictions,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
