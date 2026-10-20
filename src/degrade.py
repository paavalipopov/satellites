"""Stages 2 and 3 — building the cube, and degrading it.

The ordering matters and is easy to get wrong.  There are THREE degradations,
not one, and the blur sits *between* the two noise terms:

    stage 2a  blend      pristine <-> irradiated, one weight w per scene
    stage 2b  spectral noise   one draw per MATERIAL, shared by all its pixels
    stage 2c  paint      copy each material's spectrum down the wavelength axis
    stage 3a  blur       per-band Gaussian, sigma grows with wavelength
    stage 3b  sensor noise     independent per pixel and per band

Two cubes come out.  The clean one is painted from the *un-noised* spectra and
is the regression target for a reconstruction model; the degraded one is what
the classifier sees.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig


def material_noise(spectrum: np.ndarray, cfg: SimConfig,
                   rng: np.random.Generator) -> np.ndarray:
    """Stage 2b — one noise draw per material, SHARED by every pixel of it.

    sigma is proportional to the material's own spectral variability:
        s_noisy = s + N(0, (scale * std(s))^2)

    Because the draw is shared across the region, this term does NOT average
    away when you pool pixels — which is why spatial pooling saturates instead
    of driving accuracy to 1.
    """
    sd = float(np.std(spectrum))
    if sd <= 0:
        return spectrum.copy()
    return spectrum + rng.normal(0.0, cfg.material_noise_scale * sd, spectrum.shape)


def paint_cube(mask: np.ndarray, spectra: dict[int, np.ndarray], n_bands: int) -> np.ndarray:
    """Stage 2c — extrude the 2-D mask into a 3-D cube.

    This is the step that surprises people: there is only ONE geometry render.
    Each pixel gets its material's whole spectrum copied along the wavelength
    axis, so before blurring every band holds the *identical spatial pattern* —
    only the constant value inside each region differs from band to band.
    Background stays exactly zero.
    """
    h, w = mask.shape
    cube = np.zeros((n_bands, h, w), float)
    for mid, spec in spectra.items():
        sel = mask == mid
        if sel.any():
            cube[:, sel] = spec[:, None]
        
    return cube


def blur_cube(cube: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Stage 3a — per-band Gaussian blur, sigma growing with wavelength.

    This is the ONLY mechanism in the whole simulator that couples the spectral
    and spatial axes.  Everything else treats them independently, so a network
    cannot learn any spectral-spatial interaction beyond "redder is blurrier".

    Uses cv2 with BORDER_REFLECT_101, which matches scipy's mode="mirror".
    """
    import cv2
    sig = cfg.psf_sigma_px
    out = np.empty_like(cube)
    for j in range(cube.shape[0]):
        out[j] = cv2.GaussianBlur(cube[j], (0, 0), sigmaX=float(sig[j]),
                                  sigmaY=float(sig[j]),
                                  borderType=cv2.BORDER_REFLECT_101)
    return out


def sensor_noise(cube: np.ndarray, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Stage 3b — additive white Gaussian, independent per pixel and per band.

    The manuscript describes this as modelling "atmospheric scattering and
    absorption".  It does not: atmosphere is MULTIPLICATIVE and spectrally
    structured (Rayleigh ~ lambda^-4, the O2 A-band at 760 nm, water vapour at
    720 and 820 nm).  Additive white noise reproduces none of that, so a model
    trained here has never seen the dominant real-world spectral distortion.
    Adding a transmission term is the single biggest fidelity gap to close.
    """
    if cfg.sensor_noise_std <= 0:
        return cube.copy()
    return cube + rng.normal(0.0, cfg.sensor_noise_std, cube.shape)
