"""Synthetic HSI data generation: config, spectral library, geometry, pipeline."""
from .config import SimConfig
from .spectral_library import (GROUPS, ID_TO_NAME, MATERIAL_NAMES, N_CLASSES, NAME_TO_ID,
                               InvalidBand, SpectralLibrary, SpectralLibraryWarning,
                               UnmeasuredBand)
from .pipeline import Sample, generate_batch, generate_sample
