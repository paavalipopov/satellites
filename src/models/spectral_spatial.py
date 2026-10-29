"""Material segmentation model from the paper (Acta Astronautica submission).

Per-pixel classifier: [B, 36, H, W] reflectance cube -> [B, 47, H, W] logits
(0 = background, 1-46 = materials). Gathered from yao/src/models.py; module and
parameter names match the paper's checkpoints, which `pretrained()` loads.
"""
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Paper checkpoints (TorchScript; keys = the paper's table rows),
# copied from yao/resluts_n_model/yes_id/
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "assets/weights/spectral_spatial"
WEIGHTS = {
    "more_scenes": WEIGHTS_DIR / "model_nozoom_with_shapes.pt",     # satellites + shapes + patches
    "satellite_only": WEIGHTS_DIR / "model_nozoom_no_shapes.pt",
}


class SpectralEncoder(nn.Module):
    """Two 1x1 convs: mixes bands within each pixel, no spatial context."""

    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.encoder(x)


class SEBlock(nn.Module):
    """Squeeze-and-excitation: per-channel weights from the global mean."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class ResidualBlock(nn.Module):
    """Two 3x3 convs with GroupNorm(1, C), added back to the input."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class SpectralSpatialHSIModel(nn.Module):
    """Spectral encoder -> SE attention -> 2 residual blocks -> 2-conv decoder."""

    def __init__(self, in_channels=36, num_classes=47):
        super().__init__()
        self.spectral = SpectralEncoder(in_channels, hidden_channels=128)
        self.attention = SEBlock(128)
        self.spatial = nn.Sequential(ResidualBlock(128), ResidualBlock(128))

        # No activation between the two convs, as in the checkpoints
        # (the paper's table lists a GELU here).
        self.decoder1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.spectral(x)
        x = self.attention(x)
        x = self.spatial(x)
        x = self.decoder1(x)
        return self.decoder2(x)

    @classmethod
    def pretrained(cls, name="more_scenes"):
        """The model with trained weights; `name` is a key of WEIGHTS."""
        model = cls()
        model.load_state_dict(torch.jit.load(str(WEIGHTS[name]), map_location="cpu").state_dict())
        return model


def default_HPs(cfg):
    runtime = cfg.get("runtime") or {}
    return OmegaConf.create(
        {
            "in_channels": runtime.get("n_channels") or 36,
            "num_classes": runtime.get("n_classes") or 47,
        }
    )


MODEL = SpectralSpatialHSIModel
