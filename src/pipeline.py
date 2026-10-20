"""The pipeline, end to end, with every intermediate kept for inspection.

    stage 0   spectral library      measured DHR resampled onto the grid
    stage 1   scene mask            geometry -> material index per pixel
    stage 2a  blend                 pristine <-> irradiated, one w per scene
    stage 2b  spectral noise        one draw per material, shared by its pixels
    stage 2c  paint                 extrude the mask into a cube
    stage 3a  blur                  per-band Gaussian, sigma grows with lambda
    stage 3b  sensor noise          independent per pixel and band

`generate_sample` returns a `Sample` holding all of it.  No file I/O — if you
add saving later, write cube and mask into ONE .npz (or a temp file plus
os.replace).  The original wrote them as two separate .npy files into a ring
buffer being overwritten at ~4800 writes/s, so a DataLoader could read a cube
from one scene and the mask from another; shapes always matched, so it entered
training as silent label noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import SimConfig
from .degrade import blur_cube, material_noise, paint_cube, sensor_noise
from .scenes import patches_scene, random_scene, satellite_scene, shapes_scene
from .spectral_library import ID_TO_NAME, SpectralLibrary


@dataclass
class Sample:
    """Everything the pipeline produced, stage by stage."""
    cfg: SimConfig
    seed: int
    meta: dict[str, Any]

    mask: np.ndarray                        # [H, W] uint8   <- the LABEL
    blend_weight: float                     # w in [0, 1]
    material_ids: list[int]
    spectra_clean: dict[int, np.ndarray]    # stage 2a, per material
    spectra_noisy: dict[int, np.ndarray]    # stage 2b, per material
    cube_clean: np.ndarray                  # stage 2c from clean spectra
    cube_noisy_spectra: np.ndarray          # stage 2c from noised spectra
    cube_blurred: np.ndarray                # stage 3a
    cube: np.ndarray                        # stage 3b  <- the MODEL INPUT

    @property
    def wavelengths_nm(self) -> np.ndarray:
        return self.cfg.wavelengths_nm

    @property
    def material_names(self) -> dict[int, str]:
        return {i: ID_TO_NAME[i] for i in self.material_ids}

    @property
    def foreground_fraction(self) -> float:
        return float((self.mask != 0).mean())

    @property
    def object_span_px(self) -> int:
        ys, xs = np.where(self.mask != 0)
        if len(ys) == 0:
            return 0
        return int(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1)

    @property
    def provenance(self) -> dict[str, Any]:
        """Stamp this into anything you keep.  Losing it is how 20 of the
        original's 41 training runs became unreproducible."""
        return {"config_fingerprint": self.cfg.fingerprint(), "seed": self.seed,
                "scene_kind": self.meta.get("kind"),
                "lambda_nm": [self.cfg.lambda_min_nm, self.cfg.lambda_max_nm],
                "n_bands": self.cfg.n_bands,
                "sensor_noise_std": self.cfg.sensor_noise_std}

    def summary(self) -> str:
        L = self.wavelengths_nm
        return (
            f"Sample(seed={self.seed}, kind={self.meta.get('kind')})\n"
            f"  grid            {self.cfg.n_bands} bands, {L[0]:.1f}-{L[-1]:.1f} nm "
            f"(step {L[1]-L[0]:.4f} nm)\n"
            f"  mask            {self.mask.shape} uint8, ids {self.material_ids}\n"
            f"  cube            {self.cube.shape} {self.cube.dtype}  <- model input\n"
            f"  foreground      {self.foreground_fraction:.1%}   object span "
            f"{self.object_span_px} px\n"
            f"  blend weight    w = {self.blend_weight:.3f} (shared by all materials)\n"
            f"  psf sigma       {self.cfg.psf_sigma_px[0]:.3f} -> "
            f"{self.cfg.psf_sigma_px[-1]:.3f} px\n"
            f"  sensor noise    sigma = {self.cfg.sensor_noise_std}\n"
            f"  value range     clean [{self.cube_clean.min():.3f}, {self.cube_clean.max():.3f}]"
            f"  -> input [{self.cube.min():.3f}, {self.cube.max():.3f}]\n"
            f"  materials       " + ", ".join(f"{i}:{n}" for i, n in self.material_names.items())
        )


_SCENE_BY_NAME = {"satellite": satellite_scene, "shapes": shapes_scene,
                  "patches": patches_scene, "random": random_scene}


def generate_sample(cfg: SimConfig, seed: int, library: SpectralLibrary,
                    scene: str = "satellite", image_size: int | None = None) -> Sample:
    """One sample, with every intermediate retained.

    `scene` is "satellite" | "shapes" | "patches" | "random"; "random" draws
    from cfg.scene_probs, which is how the original mixed them (30/30/40).
    """
    rng = np.random.default_rng(seed)

    # ---- stage 1: geometry -> material index per pixel -------------------
    mask, meta = _SCENE_BY_NAME[scene](cfg, rng, image_size=image_size, library=library)
    ids = [int(i) for i in np.unique(mask) if i != 0]

    # ---- stage 2a/2b: one spectrum per material --------------------------
    # ONE blend weight for the entire scene: every part of the spacecraft is
    # assumed to have aged by exactly the same amount.
    w_scene = float(rng.random())
    clean, noisy = {}, {}
    for mid in ids:
        w = float(rng.random()) if cfg.blend_weight_per_material else w_scene
        s = library.blend(ID_TO_NAME[mid], w)
        clean[mid] = s
        noisy[mid] = material_noise(s, cfg, rng)

    # ---- stage 2c: paint -------------------------------------------------
    cube_clean = paint_cube(mask, clean, cfg.n_bands)
    cube_noisy = paint_cube(mask, noisy, cfg.n_bands)

    # ---- stage 3: degrade ------------------------------------------------
    blurred = blur_cube(cube_noisy, cfg)
    cube = sensor_noise(blurred, cfg, rng)

    return Sample(cfg=cfg, seed=seed, meta=meta, mask=mask, blend_weight=w_scene,
                  material_ids=ids, spectra_clean=clean, spectra_noisy=noisy,
                  cube_clean=cube_clean, cube_noisy_spectra=cube_noisy,
                  cube_blurred=blurred, cube=cube)


def generate_batch(cfg: SimConfig, library: SpectralLibrary, n: int,
                   seed0: int = 0, scene: str = "random") -> list[Sample]:
    return [generate_sample(cfg, seed0 + i, library, scene=scene) for i in range(n)]
