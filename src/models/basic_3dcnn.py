"""3D-CNN baseline from the later experiments (yao/train_yes_id_3dcnn_proto_srr.py).

Treats the cube as a volume: 3D convs over (band, y, x), a mean over bands, then
a small 2D head gives per-pixel logits. Module names match that script's
checkpoints, which `pretrained()` loads.
"""
from pathlib import Path

import torch
import torch.nn as nn

# Best-validation checkpoint (val mIoU 0.23), copied from
# yao/resluts_n_model/3dcnn/Pristine_basic_3dcnn_seed33/
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "assets/weights/basic_3dcnn"
WEIGHTS = {"seed33": WEIGHTS_DIR / "best_ckpt.pt"}


class Basic3DCNN(nn.Module):
    """Three 3D convs -> mean over bands -> 3x3 conv + 1x1 classifier."""

    def __init__(self, in_channels=36, num_classes=47, hidden_channels=64):
        super().__init__()
        # in_channels is unused (any number of bands works); kept so both
        # models take the same arguments
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.conv3d(x.unsqueeze(1))      # [B, 1, bands, H, W] -> [B, hidden, bands, H, W]
        x = x.mean(dim=2)                    # average over bands
        return self.head(x)

    @classmethod
    def pretrained(cls, name="seed33"):
        """The model with trained weights; `name` is a key of WEIGHTS."""
        model = cls()
        model.load_state_dict(torch.load(WEIGHTS[name], map_location="cpu")["model"])
        return model
