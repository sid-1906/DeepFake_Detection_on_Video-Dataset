#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts"),
)

from multimodal_model import MultimodalDeepfakeDetector


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "multimodal_best.pt"
)

YU_NET = (
    PROJECT_ROOT
    / "model_store"
    / "face_detection"
    / "face_detection_yunet_2023mar.onnx"
)

IMAGE_SIZE = 224
SEQUENCE_LENGTH = 8
FACE_CONFIDENCE = 0.6


# ============================================================
# VIDEO FRAME SAMPLING
# ============================================================

def sample_frame_indices(
    frame_count: int,
    number: int,
):
    """
    Select evenly distributed frames.

    This follows the same sampling strategy used during
    dataset preparation.
    """

    if frame_count <= 0:
        return []

    if frame_count <= number:
        return list(range(frame_count))

    indices = []

    for i in range(number):

        position = (
            (i + 0.5)
            / number
        )

        index = int(
            position * frame_count
        )

        index = min(
            index,
            frame_count - 1,
        )

        indices.append(index)

    return sorted(set(indices))


# ============================================================
# YUNET
# ============================================================

def create_detector():

    if not YU_NET.exists():

        raise FileNotFoundError(
            f"YuNet model not found:\n{YU_NET}"
        )

    detector = cv2.FaceDetectorYN.create(
        str(YU_NET),
        "",
        (320, 320),
        FACE_CONFIDENCE,
        0.3,
        5000,
    )

    return detector


# ============================================================
# FACE EXTRACTION
# ============================================================

def crop_primary_face(
    frame,
    detector,
):
    """
    Detect the largest face and create the same type
    of 1.25x expanded square crop used during training.
    """

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)

    if faces is None:
        return None, None

    if len(faces) == 0:
        return None, None

    # Select largest detected face.
    best = max(
        faces,
        key=lambda face:
            float(face[2] * face[3]),
    )

    x, y, w, h = best[:4]

    x = max(
        0,
        int(x),
    )

    y = max(
        0,
        int(y),
    )

    w = int(w)
    h = int(h)

    if w <= 0 or h <= 0:
        return None, None

    x2 = min(
        width,
        x + w,
    )

    y2 = min(
        height,
        y + h,
    )

    if x2 <= x or y2 <= y:
        return None, None

    # --------------------------------------------------------
    # Expand crop by 1.25x.
    # --------------------------------------------------------

    cx = (
        x + x2
    ) / 2.0

    cy = (
        y + y2
    ) / 2.0

    side = max(
        x2 - x,
        y2 - y,
    ) * 1.25

    left = int(
        cx - side / 2
    )

    top = int(
        cy - side / 2
    )

    right = int(
        cx + side / 2
    )

    bottom = int(
        cy + side / 2
    )

    # --------------------------------------------------------
    # Reflection padding.
    # --------------------------------------------------------

    pad_left = max(
        0,
        -left,
    )

    pad_top = max(
        0,
        -top,
    )

    pad_right = max(
        0,
        right - width,
    )

    pad_bottom = max(
        0,
        bottom - height,
    )

    if (
        pad_left
        or pad_top
        or pad_right
        or pad_bottom
    ):

        frame = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )

        left += pad_left
        right += pad_left

        top += pad_top
        bottom += pad_top

    crop = frame[
        top:bottom,
        left:right,
    ]

    if crop.size == 0:
        return None, None

    crop = cv2.resize(
        crop,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    confidence = float(
        best[-1]
    )

    return crop, confidence


# ============================================================
# RGB PREPROCESSING
# ============================================================

def preprocess_rgb(
    crop,
):
    """
    Convert BGR OpenCV image into the exact RGB tensor
    representation used by the training dataset.

    Output:
        [3, 224, 224]
    """

    image = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    tensor = torch.from_numpy(
        image.copy()
    ).permute(
        2,
        0,
        1,
    ).float() / 255.0

    return tensor


# ============================================================
# FREQUENCY FEATURES
# ============================================================

def frequency_features(
    rgb,
):
    """
    Produce the same two frequency channels used during
    training:

        channel 0 = normalized log FFT magnitude
        channel 1 = normalized high-frequency residual

    Input:
        [3, H, W]

    Output:
        [2, H, W]
    """

    gray = (
        0.299 * rgb[0]
        + 0.587 * rgb[1]
        + 0.114 * rgb[2]
    )

    # --------------------------------------------------------
    # Fourier transform
    # --------------------------------------------------------

    spectrum = torch.fft.fft2(
        gray
    )

    spectrum = torch.fft.fftshift(
        spectrum
    )

    magnitude = torch.log1p(
        torch.abs(spectrum)
    )

    magnitude = (
        magnitude
        - magnitude.mean()
    ) / (
        magnitude.std()
        + 1e-6
    )

    # --------------------------------------------------------
    # High-frequency residual
    # --------------------------------------------------------

    blur = torch.nn.functional.avg_pool2d(
        gray[None, None],
        kernel_size=7,
        stride=1,
        padding=3,
    )[0, 0]

    residual = (
        gray - blur
    )

    residual = (
        residual
        - residual.mean()
    ) / (
        residual.std()
        + 1e-6
    )

    return torch.stack(
        [
            magnitude,
            residual,
        ],
        dim=0,
    )


# ============================================================
# OPTICAL FLOW
# ============================================================

def motion_features(
    previous,
    current,
):
    """
    Calculate Farneback optical flow between two consecutive
    face crops.

    Output:
        [2, H, W]

    Channel 0:
        horizontal motion (dx)

    Channel 1:
        vertical motion (dy)
    """

    previous_np = (
        previous
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).astype(
        np.uint8
    )

    current_np = (
        current
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).astype(
        np.uint8
    )

    previous_gray = cv2.cvtColor(
        previous_np,
        cv2.COLOR_RGB2GRAY,
    )

    current_gray = cv2.cvtColor(
        current_np,
        cv2.COLOR_RGB2GRAY,
    )

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Same clipping used during training.
    flow = np.clip(
        flow,
        -20.0,
        20.0,
    )

    flow = torch.from_numpy(
        flow.copy()
    ).permute(
        2,
        0,
        1,
    ).float()

    flow = flow / 20.0

    return flow


# ============================================================
# LOAD VIDEO
# ============================================================

def load_video_faces(
    video_path,
    detector,
):
    """
    Read the video, sample eight frames, detect the largest
    face in every sampled frame, and return RGB face tensors.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if frame_count <= 0:

        cap.release()

        raise RuntimeError(
            "Video contains no readable frames."
        )

    indices = sample_frame_indices(
        frame_count,
        SEQUENCE_LENGTH,
    )

    if len(indices) < SEQUENCE_LENGTH:

        cap.release()

        raise RuntimeError(
            f"Video has only {len(indices)} "
            f"usable sampled frames. "
            f"The model requires exactly "
            f"{SEQUENCE_LENGTH} frames."
        )

    rgb_frames = []

    frame_information = []

    print()
    print("=" * 70)
    print("VIDEO INFORMATION")
    print("=" * 70)

    print(
        f"File:              {video_path}"
    )

    print(
        f"Frame count:       {frame_count}"
    )

    print(
        f"FPS:               {fps:.3f}"
    )

    print(
        f"Resolution:        {width} x {height}"
    )

    print(
        f"Sampled frames:    {indices}"
    )

    print()
    print("FACE EXTRACTION")
    print("-" * 70)

    for position, frame_index in enumerate(
        indices,
        start=1,
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        ok, frame = cap.read()

        if not ok:

            cap.release()

            raise RuntimeError(
                f"Could not read sampled "
                f"frame {frame_index}."
            )

        crop, confidence = crop_primary_face(
            frame,
            detector,
        )

        if crop is None:

            cap.release()

            raise RuntimeError(
                f"No face detected in "
                f"sampled frame "
                f"{frame_index}."
            )

        rgb = preprocess_rgb(
            crop
        )

        rgb_frames.append(
            rgb
        )

        timestamp = (
            frame_index / fps
            if fps > 0
            else 0.0
        )

        frame_information.append(
            {
                "position": position,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "face_confidence": confidence,
            }
        )

        print(
            f"Frame {position}/"
            f"{SEQUENCE_LENGTH} | "
            f"index={frame_index:6d} | "
            f"time={timestamp:8.3f}s | "
            f"face_confidence={confidence:.4f}"
        )

    cap.release()

    return (
        rgb_frames,
        frame_information,
        {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
        },
    )


# ============================================================
# BUILD MODEL INPUTS
# ============================================================

def build_inputs(
    rgb_frames,
):
    """
    Convert eight RGB frames into:

        RGB:
            [1, 8, 3, 224, 224]

        Frequency:
            [1, 8, 2, 224, 224]

        Motion:
            [1, 8, 2, 224, 224]
    """

    if len(rgb_frames) != SEQUENCE_LENGTH:

        raise RuntimeError(
            "Incorrect number of frames."
        )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = torch.stack(
        rgb_frames
    )

    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    frequency = torch.stack(
        [
            frequency_features(frame)
            for frame in rgb_frames
        ]
    )

    # --------------------------------------------------------
    # Motion
    # --------------------------------------------------------

    zero_motion = torch.zeros(
        2,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=torch.float32,
    )

    motion_frames = [
        zero_motion
    ]

    for i in range(
        1,
        len(rgb_frames),
    ):

        motion_frames.append(
            motion_features(
                rgb_frames[i - 1],
                rgb_frames[i],
            )
        )

    motion = torch.stack(
        motion_frames
    )

    # Add batch dimension.
    rgb = rgb.unsqueeze(0)

    frequency = frequency.unsqueeze(0)

    motion = motion.unsqueeze(0)

    return (
        rgb,
        frequency,
        motion,
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(
    device,
):
    """
    Load the exact trained architecture and checkpoint.
    """

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    feature_dim = int(
        checkpoint.get(
            "feature_dim",
            256,
        )
    )

    sequence_length = int(
        checkpoint.get(
            "sequence_length",
            8,
        )
    )

    pretrained_rgb = bool(
        checkpoint.get(
            "pretrained_rgb",
            True,
        )
    )

    if sequence_length != SEQUENCE_LENGTH:

        raise RuntimeError(
            "Checkpoint sequence length "
            f"is {sequence_length}, but "
            f"inference expects "
            f"{SEQUENCE_LENGTH}."
        )

    model = MultimodalDeepfakeDetector(
        feature_dim=feature_dim,
        pretrained_rgb=pretrained_rgb,
        sequence_length=sequence_length,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict(
    model,
    rgb,
    frequency,
    motion,
    device,
):
    """
    Run one video through the trained model.
    """

    rgb = rgb.to(
        device,
        non_blocking=True,
    )

    frequency = frequency.to(
        device,
        non_blocking=True,
    )

    motion = motion.to(
        device,
        non_blocking=True,
    )

    logits = model(
        rgb,
        frequency,
        motion,
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    real_probability = float(
        probabilities[0].item()
    )

    fake_probability = float(
        probabilities[1].item()
    )

    predicted_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    return (
        predicted_class,
        real_probability,
        fake_probability,
        logits[0].detach().cpu(),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the trained multimodal "
            "deepfake detector on a real video."
        )
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to input video.",
    )

    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path for JSON result.",
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    ).expanduser().resolve()

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video does not exist:\n{video_path}"
        )

    if not video_path.is_file():

        raise RuntimeError(
            f"Input is not a file:\n{video_path}"
        )

    print("=" * 70)
    print("MULTIMODAL DEEPFAKE VIDEO INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
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

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = load_model(
        device
    )

    print(
        "Checkpoint loaded successfully."
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Feature dimension:",
        checkpoint.get(
            "feature_dim",
            256,
        ),
    )

    print(
        "Sequence length:",
        checkpoint.get(
            "sequence_length",
            8,
        ),
    )

    print(
        "Pretrained RGB:",
        checkpoint.get(
            "pretrained_rgb",
            True,
        ),
    )

    if "best_macro_f1" in checkpoint:

        print(
            "Best validation Macro F1:",
            f"{float(checkpoint['best_macro_f1']):.4f}",
        )

    # --------------------------------------------------------
    # Face detector
    # --------------------------------------------------------

    detector = create_detector()

    # --------------------------------------------------------
    # Video preprocessing
    # --------------------------------------------------------

    (
        rgb_frames,
        frame_information,
        video_information,
    ) = load_video_faces(
        video_path,
        detector,
    )

    # --------------------------------------------------------
    # Tensor construction
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING MULTIMODAL FEATURES")
    print("=" * 70)

    (
        rgb,
        frequency,
        motion,
    ) = build_inputs(
        rgb_frames
    )

    print(
        "RGB tensor:       ",
        tuple(rgb.shape),
    )

    print(
        "Frequency tensor: ",
        tuple(frequency.shape),
    )

    print(
        "Motion tensor:    ",
        tuple(motion.shape),
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL INFERENCE")
    print("=" * 70)

    (
        predicted_class,
        real_probability,
        fake_probability,
        logits,
    ) = predict(
        model,
        rgb,
        frequency,
        motion,
        device,
    )

    if predicted_class == 0:

        prediction = "REAL"

    else:

        prediction = "FAKE"

    confidence = max(
        real_probability,
        fake_probability,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Prediction:        {prediction}"
    )

    print(
        f"Real probability:  {real_probability:.6f}"
    )

    print(
        f"Fake probability:  {fake_probability:.6f}"
    )

    print(
        f"Confidence:        {confidence:.6f}"
    )

    print()
    print(
        "Interpretation:"
    )

    if prediction == "FAKE":

        print(
            "The trained model classified the "
            "sampled facial sequence as FAKE."
        )

    else:

        print(
            "The trained model classified the "
            "sampled facial sequence as REAL."
        )

    print()
    print(
        "Important: the probabilities are model "
        "outputs and should not be interpreted "
        "as proof that a video is authentic or manipulated."
    )

    # --------------------------------------------------------
    # Optional JSON output
    # --------------------------------------------------------

    if args.save_json:

        import json

        result = {
            "video": str(video_path),
            "prediction": prediction,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "confidence": confidence,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_epoch": checkpoint.get(
                "epoch"
            ),
            "feature_dim": checkpoint.get(
                "feature_dim",
                256,
            ),
            "sequence_length": checkpoint.get(
                "sequence_length",
                8,
            ),
            "video_information": video_information,
            "sampled_frames": frame_information,
        }

        output_path = Path(
            args.save_json
        ).expanduser().resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
            )

        print()
        print(
            "JSON result saved:",
            output_path,
        )

    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
