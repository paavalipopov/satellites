"""Per-pixel material classifiers: [B, bands, H, W] cube -> [B, classes, H, W] logits."""
from .basic_3dcnn import Basic3DCNN
from .spectral_spatial import SpectralSpatialHSIModel
