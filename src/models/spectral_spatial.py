"""Material segmentation model from the paper (Acta Astronautica submission).

Per-pixel classifier: [B, 36, H, W] reflectance cube -> [B, 47, H, W] logits
(0 = background, 1-46 = materials). Gathered from yao/src/models.py; module and
parameter names match the paper's checkpoints, which `pretrained()` loads.
"""
import math
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


INPUT_NORMS = ("none", "max", "p99", "flux", "band_flux", "pixel")


class SpectralSpatialHSIModel(nn.Module):
    """Spectral encoder -> SE attention -> 2 residual blocks -> 2-conv decoder.

    input_norm first divides every input by a number taken from the input itself, so overall
    brightness drops out (what notebook 03's "max = 1" rule does by hand, here per input and
    inside the model). The default "none" is the paper's model; the layers and their weights are
    the same for every setting, so any checkpoint loads into all of them.

        none       no division (the paper's model)
        max        the brightest value of the whole input
        p99        its `norm_quantile` (0.99) quantile: one hot pixel can no longer set the scale
        flux       the mean of all values (total flux per pixel)
        band_flux  the mean of each band, separately: also takes out a smooth curve over the bands
        pixel      the brightest band of each pixel: keeps only each pixel's spectral SHAPE

    max, p99 and flux are one number per input, so they keep the brightness differences between
    the pixels (a dark material stays darker than a bright one). band_flux and pixel are not, and
    pixel also turns every empty background pixel into a full-strength one.
    """

    def __init__(self, in_channels=36, num_classes=47, input_norm="none", norm_eps=1e-8,
                 norm_quantile=0.99):
        super().__init__()
        if input_norm not in INPUT_NORMS:
            raise ValueError(f"input_norm={input_norm!r}: one of {INPUT_NORMS}")
        if not 0.0 < norm_quantile <= 1.0:
            raise ValueError(f"norm_quantile={norm_quantile}: in (0, 1]")
        self.input_norm, self.norm_eps, self.norm_quantile = input_norm, norm_eps, norm_quantile
        self.spectral = SpectralEncoder(in_channels, hidden_channels=128)
        self.attention = SEBlock(128)
        self.spatial = nn.Sequential(ResidualBlock(128), ResidualBlock(128))

        # No activation between the two convs, as in the checkpoints
        # (the paper's table lists a GELU here).
        self.decoder1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def input_scale(self, x):
        """The number(s) the input is divided by, broadcastable against x [B, C, H, W]
        (None when input_norm is "none")."""
        mode = self.input_norm
        if mode == "max":
            scale = x.amax(dim=(1, 2, 3), keepdim=True)
        elif mode == "p99":
            flat = x.flatten(1)
            k = max(1, math.ceil(self.norm_quantile * flat.shape[1]))
            scale = flat.kthvalue(k, dim=1).values.view(-1, 1, 1, 1)
        elif mode == "flux":
            scale = x.mean(dim=(1, 2, 3), keepdim=True)
        elif mode == "band_flux":
            scale = x.mean(dim=(2, 3), keepdim=True)
        elif mode == "pixel":
            scale = x.amax(dim=1, keepdim=True)
        else:
            return None
        return scale.clamp_min(self.norm_eps)

    def forward(self, x, scale=None):
        """`scale`: divide by this instead of the input's own number, e.g. `model.input_scale(whole_image)`
        so that an empty window is not blown up to full strength."""
        if scale is None:
            scale = self.input_scale(x)
        if scale is not None:
            x = x / scale
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
            "input_norm": "none",       # none | max | p99 | flux | band_flux | pixel (see the class)
            "norm_eps": 1e-8,           # floor under the divisor, so an all-zero input stays zero
            "norm_quantile": 0.99,      # for p99
        }
    )


MODEL = SpectralSpatialHSIModel
