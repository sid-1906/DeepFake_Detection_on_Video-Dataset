#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        output_dim=256,
        pretrained=False,
    ):
        super().__init__()

        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = convnext_tiny(weights=weights)

        if input_channels != 3:
            old_conv = backbone.features[0][0]

            new_conv = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            nn.init.kaiming_normal_(
                new_conv.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            backbone.features[0][0] = new_conv

        # Keep ConvNeXt feature extraction but explicitly flatten
        # the final [B, 768, 1, 1] feature map.
        backbone.classifier = nn.Identity()

        self.backbone = backbone

        self.flatten = nn.Flatten(1)

        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        return self.projection(x)


class CrossBranchFusion(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, rgb, frequency, motion):
        """
        Inputs:
            [B,T,D]
        """

        tokens = torch.stack(
            [rgb, frequency, motion],
            dim=2,
        )

        b, t, branches, d = tokens.shape

        tokens = tokens.reshape(
            b * t,
            branches,
            d,
        )

        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
        )

        attended = self.norm1(
            tokens + attended
        )

        attended = self.norm2(
            attended + self.ffn(attended)
        )

        attended = attended.reshape(
            b,
            t,
            branches,
            d,
        )

        rgb_a = attended[:, :, 0]
        freq_a = attended[:, :, 1]
        motion_a = attended[:, :, 2]

        combined = torch.cat(
            [rgb_a, freq_a, motion_a],
            dim=-1,
        )

        gate = self.gate(combined)

        fused = (
            gate * rgb_a
            + (1.0 - gate) * (
                (freq_a + motion_a) / 2.0
            )
        )

        return fused


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        dim=256,
        heads=8,
        layers=2,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)

        return x.mean(dim=1)


class MultimodalDeepfakeDetector(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        pretrained_rgb=False,
        pretrained_frequency=False,
        pretrained_motion=False,
    ):
        super().__init__()

        self.rgb_encoder = ConvNeXtEncoder(
            input_channels=3,
            output_dim=feature_dim,
            pretrained=pretrained_rgb,
        )

        self.frequency_encoder = ConvNeXtEncoder(
            input_channels=2,
            output_dim=feature_dim,
            pretrained=pretrained_frequency,
        )

        self.motion_encoder = ConvNeXtEncoder(
            input_channels=2,
            output_dim=feature_dim,
            pretrained=pretrained_motion,
        )

        self.fusion = CrossBranchFusion(
            dim=feature_dim,
            heads=8,
        )

        self.temporal = TemporalTransformer(
            dim=feature_dim,
            heads=8,
            layers=2,
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 2),
        )

    def forward(
        self,
        rgb,
        frequency,
        motion,
    ):
        """
        Inputs:
            rgb:
                [B,T,3,H,W]

            frequency:
                [B,T,2,H,W]

            motion:
                [B,T,2,H,W]
        """

        b, t = rgb.shape[:2]

        rgb = rgb.reshape(
            b * t,
            *rgb.shape[2:],
        )

        frequency = frequency.reshape(
            b * t,
            *frequency.shape[2:],
        )

        motion = motion.reshape(
            b * t,
            *motion.shape[2:],
        )

        rgb_features = self.rgb_encoder(rgb)
        frequency_features = self.frequency_encoder(
            frequency
        )
        motion_features = self.motion_encoder(
            motion
        )

        rgb_features = rgb_features.reshape(
            b, t, -1
        )

        frequency_features = frequency_features.reshape(
            b, t, -1
        )

        motion_features = motion_features.reshape(
            b, t, -1
        )

        fused = self.fusion(
            rgb_features,
            frequency_features,
            motion_features,
        )

        temporal = self.temporal(fused)

        logits = self.classifier(temporal)

        return logits


if __name__ == "__main__":
    model = MultimodalDeepfakeDetector(
        pretrained_rgb=False,
    )

    rgb = torch.randn(
        2, 8, 3, 224, 224
    )

    frequency = torch.randn(
        2, 8, 2, 224, 224
    )

    motion = torch.randn(
        2, 8, 2, 224, 224
    )

    logits = model(
        rgb,
        frequency,
        motion,
    )

    print("Output:", logits.shape)
