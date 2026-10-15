from pathlib import Path

import cv2


MODEL_PATH = Path(
    "model_store/face_detection/"
    "face_detection_yunet_2023mar.onnx"
)

IMAGE_PATH = Path(
    "data/processed/frames/frame_000000.jpg"
)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {MODEL_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Frame not found: {IMAGE_PATH}"
        )

    detector = cv2.FaceDetectorYN.create(
        str(MODEL_PATH),
        "",
        (320, 320),
        0.6,
        0.3,
        5000,
    )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {IMAGE_PATH}"
        )

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    count = 0 if faces is None else len(faces)

    print("Image:", IMAGE_PATH)
    print("Resolution:", f"{width}x{height}")
    print("Faces detected:", count)

    if faces is not None:
        for index, face in enumerate(faces):
            print(f"\nFace {index}:")
            print(face)


if __name__ == "__main__":
    main()
