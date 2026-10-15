#!/usr/bin/env python3

from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FFPPMultimodalDataset(Dataset):
    """
    FaceForensics++ multimodal dataset.

    Each dataset item contains exactly `sequence_length` face frames.

    Returns:
        rgb:
            [T, 3, H, W]

        frequency:
            [T, 2, H, W]

        motion:
            [T, 2, H, W]

        label:
            scalar tensor
            0 = REAL
            1 = FAKE

        metadata:
            dictionary containing source information
    """

    def __init__(
        self,
        manifest_path,
        image_size=224,
        sequence_length=8,
    ):
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.is_absolute():
            self.manifest_path = (
                PROJECT_ROOT / self.manifest_path
            )

        self.image_size = int(image_size)
        self.sequence_length = int(sequence_length)

        if self.sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero."
            )

        if self.image_size <= 0:
            raise ValueError(
                "image_size must be greater than zero."
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest does not exist: "
                f"{self.manifest_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.records = json.load(f)

        if not isinstance(self.records, list):
            raise RuntimeError(
                "Dataset manifest must contain a JSON list."
            )

        if not self.records:
            raise RuntimeError(
                "Dataset manifest is empty."
            )

        # --------------------------------------------------
        # Keep only records with enough frames.
        # --------------------------------------------------

        original_count = len(self.records)

        valid_records = []

        for record in self.records:

            if not isinstance(record, dict):
                continue

            frames = record.get("frames", [])

            if not isinstance(frames, list):
                continue

            if len(frames) >= self.sequence_length:
                valid_records.append(record)

        self.records = valid_records

        self.skipped_records = (
            original_count - len(self.records)
        )

        if not self.records:
            raise RuntimeError(
                "Dataset has no records with enough frames."
            )

    def __len__(self):
        return len(self.records)

    # ======================================================
    # PATH / FRAME HANDLING
    # ======================================================

    def _resolve_path(self, path):
        """
        Resolve a dataset path safely.

        Accepts:
            - pathlib.Path
            - string
            - frame dictionary containing "image"
            - frame dictionary containing "path"

        This is important because the manifest stores
        frame entries as dictionaries.
        """

        # ----------------------------------------------
        # Frame dictionary
        # ----------------------------------------------

        if isinstance(path, dict):

            # Normal manifest format.
            if "image" in path:
                path = path["image"]

            # Fallback format.
            elif "path" in path:
                path = path["path"]

            else:
                raise RuntimeError(
                    "Frame dictionary does not contain "
                    "'image' or 'path': "
                    f"{path}"
                )

        # ----------------------------------------------
        # Path object
        # ----------------------------------------------

        path = Path(path)

        # ----------------------------------------------
        # Absolute path
        # ----------------------------------------------

        if path.is_absolute():
            return path

        # ----------------------------------------------
        # Manifest paths are normally project-relative.
        # ----------------------------------------------

        return PROJECT_ROOT / path

    # ======================================================
    # IMAGE LOADING
    # ======================================================

    def _load_image(self, path):

        path = self._resolve_path(path)

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size,
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

    # ======================================================
    # FREQUENCY FEATURES
    # ======================================================

    def _frequency_features(self, rgb):
        """
        Produce two frequency-domain channels.

        Channel 0:
            normalized log FFT magnitude

        Channel 1:
            normalized high-frequency residual

        Input:
            [3,H,W]

        Output:
            [2,H,W]
        """

        gray = (
            0.299 * rgb[0]
            + 0.587 * rgb[1]
            + 0.114 * rgb[2]
        )

        # ----------------------------------------------
        # FFT magnitude
        # ----------------------------------------------

        spectrum = torch.fft.fft2(gray)

        spectrum = torch.fft.fftshift(
            spectrum
        )

        magnitude = torch.log1p(
            torch.abs(spectrum)
        )

        magnitude = (
            magnitude - magnitude.mean()
        ) / (
            magnitude.std() + 1e-6
        )

        # ----------------------------------------------
        # High-frequency residual
        # ----------------------------------------------

        blur = torch.nn.functional.avg_pool2d(
            gray[None, None],
            kernel_size=7,
            stride=1,
            padding=3,
        )[0, 0]

        residual = gray - blur

        residual = (
            residual - residual.mean()
        ) / (
            residual.std() + 1e-6
        )

        return torch.stack(
            [
                magnitude,
                residual,
            ],
            dim=0,
        )

    # ======================================================
    # MOTION FEATURES
    # ======================================================

    def _motion_features(
        self,
        previous,
        current,
    ):
        """
        Calculate dense optical flow between
        consecutive face crops.

        Input:
            previous [3,H,W]
            current  [3,H,W]

        Output:
            [2,H,W]
            channel 0 = dx
            channel 1 = dy
        """

        previous_np = (
            previous
            .permute(1, 2, 0)
            .numpy()
            * 255.0
        ).astype(np.uint8)

        current_np = (
            current
            .permute(1, 2, 0)
            .numpy()
            * 255.0
        ).astype(np.uint8)

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

        # Prevent extreme optical-flow values.
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

        # Normalize to approximately [-1, 1].
        flow = flow / 20.0

        return flow

    # ======================================================
    # DATASET ITEM
    # ======================================================

    def __getitem__(self, index):

        record = self.records[index]

        if not isinstance(record, dict):
            raise RuntimeError(
                f"Invalid record at index {index}: "
                f"{record}"
            )

        # --------------------------------------------------
        # Frames
        # --------------------------------------------------

        frames = record.get("frames")

        if not isinstance(frames, list):
            raise RuntimeError(
                f"Record has invalid frames field: "
                f"{record.get('path', 'UNKNOWN')}"
            )

        # --------------------------------------------------
        # IMPORTANT:
        # Never silently accept an incomplete sequence.
        # --------------------------------------------------

        if len(frames) < self.sequence_length:
            raise RuntimeError(
                f"Record has only {len(frames)} frames: "
                f"{record.get('path', 'UNKNOWN')}"
            )

        # Exactly the temporal sequence expected by model.
        frames = frames[:self.sequence_length]

        # --------------------------------------------------
        # RGB frames
        # --------------------------------------------------

        rgb_frames = []

        for frame_index, frame in enumerate(frames):

            try:
                image = self._load_image(frame)

            except Exception as exc:

                record_path = record.get(
                    "path",
                    "UNKNOWN",
                )

                raise RuntimeError(
                    f"Failed to load frame "
                    f"{frame_index} for record "
                    f"{record_path}: {exc}"
                ) from exc

            rgb_frames.append(image)

        # --------------------------------------------------
        # Stack RGB sequence
        # --------------------------------------------------

        rgb = torch.stack(
            rgb_frames,
            dim=0,
        )

        # --------------------------------------------------
        # Frequency sequence
        # --------------------------------------------------

        frequency_frames = []

        for frame in rgb_frames:

            frequency_frames.append(
                self._frequency_features(frame)
            )

        frequency = torch.stack(
            frequency_frames,
            dim=0,
        )

        # --------------------------------------------------
        # Motion sequence
        # --------------------------------------------------

        motion_frames = []

        # First frame has no previous frame.
        zero_motion = torch.zeros(
            2,
            self.image_size,
            self.image_size,
            dtype=torch.float32,
        )

        motion_frames.append(
            zero_motion
        )

        for i in range(
            1,
            len(rgb_frames),
        ):

            motion_frames.append(
                self._motion_features(
                    rgb_frames[i - 1],
                    rgb_frames[i],
                )
            )

        motion = torch.stack(
            motion_frames,
            dim=0,
        )

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        label_name = str(
            record.get(
                "label",
                "",
            )
        ).lower().strip()

        if label_name == "fake":
            label = 1

        elif label_name == "real":
            label = 0

        else:
            raise RuntimeError(
                f"Unknown label '{label_name}' "
                f"for record: "
                f"{record.get('path', 'UNKNOWN')}"
            )

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata = {
            "path": record.get(
                "path",
                "",
            ),

            "method": record.get(
                "method",
                "",
            ),

            "source_group": record.get(
                "source_group",
                "",
            ),
        }

        # --------------------------------------------------
        # Final sample
        # --------------------------------------------------

        return {
            "rgb": rgb,

            "frequency": frequency,

            "motion": motion,

            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),

            "metadata": metadata,
        }


# ==========================================================
# DIRECT DATASET CHECK
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MULTIMODAL DATASET CHECK")
    print("=" * 70)

    dataset = FFPPMultimodalDataset(
        "data/dataset/metadata/face_samples_train.json",
        image_size=224,
        sequence_length=8,
    )

    print(
        "Manifest:",
        dataset.manifest_path,
    )

    print(
        "Dataset records:",
        len(dataset),
    )

    print(
        "Skipped records:",
        dataset.skipped_records,
    )

    sample = dataset[0]

    print()
    print("First sample")
    print("-" * 70)

    print(
        "RGB:",
        tuple(sample["rgb"].shape),
        sample["rgb"].dtype,
    )

    print(
        "Frequency:",
        tuple(sample["frequency"].shape),
        sample["frequency"].dtype,
    )

    print(
        "Motion:",
        tuple(sample["motion"].shape),
        sample["motion"].dtype,
    )

    print(
        "Label:",
        sample["label"].item(),
    )

    print(
        "Metadata:",
        sample["metadata"],
    )

    # ------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------

    expected_rgb = (
        8,
        3,
        224,
        224,
    )

    expected_frequency = (
        8,
        2,
        224,
        224,
    )

    expected_motion = (
        8,
        2,
        224,
        224,
    )

    if tuple(sample["rgb"].shape) != expected_rgb:
        raise RuntimeError(
            f"Unexpected RGB shape: "
            f"{tuple(sample['rgb'].shape)}"
        )

    if tuple(sample["frequency"].shape) != expected_frequency:
        raise RuntimeError(
            f"Unexpected frequency shape: "
            f"{tuple(sample['frequency'].shape)}"
        )

    if tuple(sample["motion"].shape) != expected_motion:
        raise RuntimeError(
            f"Unexpected motion shape: "
            f"{tuple(sample['motion'].shape)}"
        )

    print()
    print("=" * 70)
    print("DATASET CHECK: PASS")
    print("=" * 70)