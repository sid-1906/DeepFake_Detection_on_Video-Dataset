#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


# ============================================================
# Projection
# ============================================================

class ProjectionHead(nn.Module):

    def __init__(self, input_dim, output_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# RGB ENCODER
# ============================================================

class RGBEncoder(nn.Module):
    """
    Pretrained ConvNeXt-Tiny RGB encoder.

    Input:
        [N, 3, 224, 224]

    Output:
        [N, 256]
    """

    def __init__(
        self,
        output_dim=256,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = convnext_tiny(
            weights=weights
        )

        # Remove ImageNet classifier.
        backbone.classifier = nn.Identity()

        self.backbone = backbone

        # ImageNet normalization belongs ONLY
        # inside the pretrained RGB branch.
        self.register_buffer(
            "rgb_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "rgb_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        self.projection = ProjectionHead(
            input_dim=768,
            output_dim=output_dim,
        )

    def forward(self, x):

        x = (
            x - self.rgb_mean
        ) / self.rgb_std

        x = self.backbone(x)

        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(
                x,
                1,
            )

            x = torch.flatten(
                x,
                1,
            )

        else:
            x = torch.flatten(
                x,
                1,
            )

        return self.projection(x)


# ============================================================
# FREQUENCY ENCODER
# ============================================================

class FrequencyResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.conv5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(
            channels
        )

        self.activation = nn.GELU()

        hidden = max(
            channels // 4,
            8,
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):

        residual = x

        a = self.conv3(x)
        b = self.conv5(x)

        x = a + b
        x = self.norm(x)
        x = self.activation(x)

        gate = self.channel_gate(x)

        x = x * gate

        return x + residual


class FrequencyEncoder(nn.Module):

    def __init__(
        self,
        input_channels=2,
        output_dim=256,
    ):
        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.block1 = FrequencyResidualBlock(
            64
        )

        self.down1 = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.block2 = FrequencyResidualBlock(
            128
        )

        self.down2 = nn.Sequential(

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.block3 = FrequencyResidualBlock(
            256
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.projection = ProjectionHead(
            input_dim=256,
            output_dim=output_dim,
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.block1(x)

        x = self.down1(x)

        x = self.block2(x)

        x = self.down2(x)

        x = self.block3(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1,
        )

        return self.projection(x)


# ============================================================
# MOTION ENCODER
# ============================================================

class MotionSpatialEncoder(nn.Module):

    def __init__(
        self,
        input_channels=2,
        feature_dim=128,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),

            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(
                128,
                feature_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                feature_dim
            ),

            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):

        x = self.net(x)

        return torch.flatten(
            x,
            1,
        )


class MotionEncoder(nn.Module):

    def __init__(
        self,
        output_dim=256,
        spatial_dim=128,
    ):
        super().__init__()

        self.spatial = MotionSpatialEncoder(
            input_channels=2,
            feature_dim=spatial_dim,
        )

        self.temporal = nn.Sequential(

            nn.Conv1d(
                spatial_dim,
                spatial_dim,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(
                spatial_dim
            ),

            nn.GELU(),

            nn.Conv1d(
                spatial_dim,
                spatial_dim,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(
                spatial_dim
            ),

            nn.GELU(),
        )

        self.projection = ProjectionHead(
            input_dim=spatial_dim,
            output_dim=output_dim,
        )

    def forward(self, x):
        """
        x:
            [B,T,2,H,W]

        returns:
            [B,T,256]
        """

        batch_size, time, channels, height, width = (
            x.shape
        )

        x = x.reshape(
            batch_size * time,
            channels,
            height,
            width,
        )

        x = self.spatial(x)

        x = x.reshape(
            batch_size,
            time,
            -1,
        )

        # Conv1d expects [B,C,T].
        x = x.transpose(
            1,
            2,
        )

        x = self.temporal(x)

        x = x.transpose(
            1,
            2,
        )

        x = self.projection(x)

        return x


# ============================================================
# MULTIMODAL DETECTOR
# ============================================================

class MultimodalDeepfakeDetector(nn.Module):

    def __init__(
        self,
        feature_dim=256,
        pretrained_rgb=True,
        sequence_length=8,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.sequence_length = sequence_length

        self.rgb_encoder = RGBEncoder(
            output_dim=feature_dim,
            pretrained=pretrained_rgb,
        )

        self.frequency_encoder = FrequencyEncoder(
            input_channels=2,
            output_dim=feature_dim,
        )

        self.motion_encoder = MotionEncoder(
            output_dim=feature_dim,
            spatial_dim=128,
        )

        # Learn how much each modality contributes.
        self.modality_fusion = nn.Sequential(

            nn.Linear(
                feature_dim * 3,
                feature_dim * 2,
            ),

            nn.LayerNorm(
                feature_dim * 2
            ),

            nn.GELU(),

            nn.Dropout(0.15),

            nn.Linear(
                feature_dim * 2,
                feature_dim,
            ),

            nn.LayerNorm(
                feature_dim
            ),

            nn.GELU(),
        )

        # Temporal transformer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.temporal_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
            )
        )

        self.temporal_norm = nn.LayerNorm(
            feature_dim
        )

        # Attention pooling over the 8 frames.
        self.temporal_attention = nn.Sequential(

            nn.Linear(
                feature_dim,
                feature_dim // 2,
            ),

            nn.Tanh(),

            nn.Linear(
                feature_dim // 2,
                1,
            ),
        )

        # Final classifier.
        self.classifier = nn.Sequential(

            nn.Linear(
                feature_dim,
                128,
            ),

            nn.LayerNorm(128),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(
                128,
                2,
            ),
        )

    def forward(
        self,
        rgb,
        frequency,
        motion,
    ):
        """
        rgb:
            [B,T,3,H,W]

        frequency:
            [B,T,2,H,W]

        motion:
            [B,T,2,H,W]

        returns:
            [B,2]
        """

        batch_size, time = (
            rgb.shape[0],
            rgb.shape[1],
        )

        # --------------------------------------------------
        # RGB
        # --------------------------------------------------

        rgb = rgb.reshape(
            batch_size * time,
            *rgb.shape[2:],
        )

        rgb_features = self.rgb_encoder(
            rgb
        )

        rgb_features = rgb_features.reshape(
            batch_size,
            time,
            self.feature_dim,
        )

        # --------------------------------------------------
        # Frequency
        # --------------------------------------------------

        frequency = frequency.reshape(
            batch_size * time,
            *frequency.shape[2:],
        )

        frequency_features = (
            self.frequency_encoder(
                frequency
            )
        )

        frequency_features = (
            frequency_features.reshape(
                batch_size,
                time,
                self.feature_dim,
            )
        )

        # --------------------------------------------------
        # Motion
        # --------------------------------------------------

        motion_features = self.motion_encoder(
            motion
        )

        # --------------------------------------------------
        # Fuse modalities per frame
        # --------------------------------------------------

        fused = torch.cat(
            [
                rgb_features,
                frequency_features,
                motion_features,
            ],
            dim=-1,
        )

        fused = self.modality_fusion(
            fused
        )

        # --------------------------------------------------
        # Temporal modelling
        # --------------------------------------------------

        temporal = self.temporal_encoder(
            fused
        )

        temporal = self.temporal_norm(
            temporal
        )

        # --------------------------------------------------
        # Attention pooling
        # --------------------------------------------------

        attention_logits = (
            self.temporal_attention(
                temporal
            )
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        pooled = (
            temporal
            * attention_weights
        ).sum(dim=1)

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.classifier(
            pooled
        )

        return logits


if __name__ == "__main__":

    print(
        "MultimodalDeepfakeDetector"
    )

    model = MultimodalDeepfakeDetector(
        feature_dim=256,
        pretrained_rgb=False,
    )

    rgb = torch.randn(
        2,
        8,
        3,
        224,
        224,
    )

    frequency = torch.randn(
        2,
        8,
        2,
        224,
        224,
    )

    motion = torch.randn(
        2,
        8,
        2,
        224,
        224,
    )

    with torch.no_grad():

        output = model(
            rgb,
            frequency,
            motion,
        )

    print(
        "Output shape:",
        output.shape,
    )