"""Stages 1-3 of the generator, and `generate_sample`, which runs them.

    stage 0   spectral library      measured DHR on the grid (spectral_library.py)
    stage 1   scene mask            satellite / shapes / patches -> material id per pixel
    stage 2a  blend                 pristine <-> irradiated, one w per scene
    stage 2b  spectral noise        one draw per material, shared by its pixels
    stage 2c  paint                 extrude the mask into a cube
    stage 3a  blur                  per-band Gaussian, sigma grows with lambda
    stage 3b  sensor noise          independent per pixel and band

`generate_sample` returns a `Sample` holding every intermediate.  No file I/O:
if you add saving, write mask and cubes into ONE .npz (or a temp file plus
os.replace).  The original wrote three separate .npy files into a ring buffer
overwritten at ~4800 writes/s, so a reader could pair a cube with another
scene's mask; shapes always matched, so it entered training as label noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import SimConfig
from .geometry import box, concat, cylinder, icosphere, rasterize, translate
from .spectral_library import GROUPS, ID_TO_NAME, NAME_TO_ID, SpectralLibrary


# ======================================================================
# Stage 1 — scene mask: one material id per pixel, 0 = background.
# Every function takes an explicit Generator, so a seed fully determines the scene.
# ======================================================================

SHAPE_TYPES = ("circle", "ring", "rectangle", "line", "ellipse",
               "polygon", "triangle", "diamond", "cross")


def build_satellite(cfg: SimConfig, rng: np.random.Generator, library=None):
    """Layout 1: bus (two halves), connectors, antenna and solar panels.
    Returns (components, chosen_material_names).

    Materials are drawn 3-from-bus, 1-from-antenna, 1-from-wings — each part
    only ever wears materials from its own group.  That is a strong prior: the
    model can partly infer material from position rather than from spectrum.
    """
    G = library.groups if library is not None else GROUPS
    comps, chosen = [], {}
    h = 3.0
    half = h / 2.0

    bus = list(rng.choice(G["bus"], 3, replace=False))
    chosen["bus"] = bus
    comps.append((translate(cylinder(0.8, half, 20), [0, 0, -h / 2 + half / 2]), NAME_TO_ID[bus[0]]))
    comps.append((translate(cylinder(0.8, half, 20), [0, 0, +h / 2 - half / 2]), NAME_TO_ID[bus[1]]))
    connectors = concat(translate(box([2.0, 0.3, 0.3]), [+2.0, 0, 0]),
                        translate(box([2.0, 0.3, 0.3]), [-2.0, 0, 0]))
    comps.append((connectors, NAME_TO_ID[bus[2]]))

    ant = str(rng.choice(G["antenna"]))
    chosen["antenna"] = ant
    comps.append((translate(icosphere(2, 0.4), [0, 0, h / 2 + 0.8]), NAME_TO_ID[ant]))

    wing = str(rng.choice(G["wings"]))
    chosen["wings"] = wing
    t = cfg.panel_thickness
    panels = concat(translate(box([4.0, t, 2.0]), [+6.0, 0, 0]),
                    translate(box([4.0, t, 2.0]), [-6.0, 0, 0]))
    comps.append((panels, NAME_TO_ID[wing]))
    return comps, chosen


def build_satellite_v2(cfg: SimConfig, rng: np.random.Generator, library=None):
    """The SECOND spacecraft layout — the original picked between the two with
    equal probability (`idx = np.random.randint(0,2)`), so half of all satellite
    scenes look like this.  It is a substantially bigger object: a wider bus, a
    long boom, and a 3.2-radius dish.

    Note the material draws: the two bus halves come from one
    `choice(..., 2, replace=False)` but the boom is an INDEPENDENT draw, so the
    boom can share a material with a body half.
    """
    G = library.groups if library is not None else GROUPS
    comps, chosen = [], {}
    h_total, r_body = 3.5, 1.25
    h_half = h_total / 2.0

    bus = list(rng.choice(G["bus"], 2, replace=False))
    comps.append((translate(cylinder(r_body, h_half, 24), [0, 0, -h_half / 2]), NAME_TO_ID[bus[0]]))
    comps.append((translate(cylinder(r_body, h_half, 24), [0, 0, +h_half / 2]), NAME_TO_ID[bus[1]]))

    wing = str(rng.choice(G["wings"]))
    px = r_body + 0.12 + 6.0 / 2.0
    panels = concat(translate(box([6.0, cfg.panel_thickness, 3.0]), [+px, 0, 0]),
                    translate(box([6.0, cfg.panel_thickness, 3.0]), [-px, 0, 0]))
    comps.append((panels, NAME_TO_ID[wing]))

    boom_mat = str(rng.choice(G["bus"]))          # independent of `bus`
    comps.append((translate(box([0.15, 0.15, 6.0]), [0, 0, h_half + 3.0]), NAME_TO_ID[boom_mat]))

    ant = str(rng.choice(G["antenna"]))
    dish = translate(cylinder(3.2, 0.03, 64), [0, 0, h_half + 6.0])
    feed = translate(icosphere(2, 0.25), [0, 0, h_half + 5.3])
    comps.append((concat(dish, feed), NAME_TO_ID[ant]))

    # The whole assembly is re-centred on its bounding box.
    lo = np.min([v.min(0) for v, _ in [c[0] for c in comps]], axis=0)
    hi = np.max([v.max(0) for v, _ in [c[0] for c in comps]], axis=0)
    ctr = (lo + hi) / 2.0
    comps = [((v - ctr, f), mid) for (v, f), mid in comps]

    chosen = {"bus": bus, "boom": boom_mat, "antenna": ant, "wings": wing}
    return comps, chosen


def satellite_scene(cfg: SimConfig, rng: np.random.Generator, image_size=None,
                    variant: int | None = None, library=None):
    """The original chose between the two layouts with equal probability."""
    if variant is None:
        variant = int(rng.integers(0, 2))
    builder = build_satellite if variant == 0 else build_satellite_v2
    comps, chosen = builder(cfg, rng, library)
    # Original ranges.  The y (horizontal) term was commented out, so the
    # satellite only ever tips about x and spins about z.
    angles = np.array([float(rng.integers(45, 135)), 0.0, float(rng.integers(-180, 180))])
    mask = rasterize(comps, cfg, angles, image_size)
    return mask, {"kind": "satellite", "variant": variant,
                  "angles_deg": angles, "materials": chosen}


def shapes_scene(cfg: SimConfig, rng: np.random.Generator, max_regions: int = 3,
                 image_size=None, library=None):
    """A few large random regions, each a distinct material.

    `max_regions=3` is the original's value: `shape_simulator` declares it as a
    default and `generator_realtime.simulator_worker` never overrides it, even
    though `random_complex_region_mask` itself defaults to 5.
    """
    n = image_size or cfg.image_size
    pool = library.usable_ids if library is not None else np.arange(1, 47)
    img = np.zeros((n, n), np.uint8)
    used: set[int] = set()
    kinds = []

    def ri(a, b):
        return int(a) if b < a else int(rng.integers(a, b + 1))

    for _ in range(int(rng.integers(1, max_regions + 1))):
        if len(used) >= len(pool):
            break
        lab = int(rng.choice(pool))
        while lab in used:
            lab = int(rng.choice(pool))
        used.add(lab)
        kind = str(rng.choice(SHAPE_TYPES))
        kinds.append(kind)
        _draw_shape(img, kind, lab, n, ri, rng)

    return img, {"kind": "shapes", "shapes": kinds}


def _draw_shape(img, kind, lab, n, ri, rng):
    if kind == "circle":
        cv2.circle(img, (ri(0, n - 1), ri(0, n - 1)), ri(2, max(2, n // 6)), lab, -1)
    elif kind == "ring":
        cx, cy = ri(0, n - 1), ri(0, n - 1)
        outer = ri(5, max(5, n // 5)); inner = ri(1, max(1, outer - 2))
        cv2.circle(img, (cx, cy), outer, lab, -1)
        cv2.circle(img, (cx, cy), inner, 0, -1)
    elif kind == "rectangle":
        w, h = ri(2, n // 4), ri(2, n // 4)
        pts = cv2.boxPoints(((ri(w, n - w), ri(h, n - h)), (w, h), ri(0, 180)))
        cv2.fillPoly(img, [pts.astype(np.int32)], lab)
    elif kind == "line":
        cx, cy = ri(0, n - 1), ri(0, n - 1)
        L = ri(4, n // 2); a = np.deg2rad(ri(0, 180))
        dx, dy = int(L * np.cos(a)), int(L * np.sin(a))
        cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), lab, 2)
    elif kind == "ellipse":
        cv2.ellipse(img, (ri(0, n - 1), ri(0, n - 1)),
                    (ri(2, n // 5), ri(2, n // 5)), ri(0, 180), 0, 360, lab, -1)
    elif kind == "polygon":
        k = ri(4, 8)
        pts = np.column_stack([rng.integers(0, n, k), rng.integers(0, n, k)]).astype(np.int32)
        cv2.fillPoly(img, [_rot(pts, (n // 2, n // 2), ri(0, 180))], lab)
    elif kind == "triangle":
        # the original drew only 2 points here, so its "triangles" were lines
        pts = np.column_stack([rng.integers(0, n, 3), rng.integers(0, n, 3)]).astype(np.int32)
        c = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        cv2.fillPoly(img, [_rot(pts, c, ri(0, 180))], lab)
    elif kind == "diamond":
        cx, cy = ri(0, n - 1), ri(0, n - 1); s = ri(2, max(2, n // 6))
        pts = np.array([[cx, cy - s], [cx + s, cy], [cx, cy + s], [cx - s, cy]], np.int32)
        cv2.fillPoly(img, [_rot(pts, (cx, cy), ri(0, 180))], lab)
    elif kind == "cross":
        cx, cy = ri(0, n - 1), ri(0, n - 1)
        s = ri(4, max(4, n // 6)); th = max(1, s // 4)
        cr = np.zeros((n, n), np.uint8)
        cv2.rectangle(cr, (cx - th, cy - s), (cx + th, cy + s), 255, -1)
        cv2.rectangle(cr, (cx - s, cy - th), (cx + s, cy + th), 255, -1)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), ri(0, 180), 1.0)
        img[cv2.warpAffine(cr, M, (n, n)) > 0] = lab


def _rot(pts, centre, deg):
    """Rotate int32 points; cv2 rounds the result, as in the original."""
    M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), deg, 1.0)
    return cv2.transform(pts.reshape(-1, 1, 2).astype(np.int32), M)


def patches_scene(cfg: SimConfig, rng: np.random.Generator,
                  patch_sizes=(1, 2, 3, 4), n_patches=(12, 15), image_size=None,
                  library=None):
    """Small non-overlapping squares, each a distinct material."""
    n = image_size or cfg.image_size
    mask = np.zeros((n, n), np.uint8)
    pool = (library.usable_ids if library is not None else np.arange(1, 47)).astype(np.uint8)
    k = int(rng.integers(n_patches[0], n_patches[1] + 1))
    labels = rng.choice(pool, size=min(k, len(pool)), replace=False)

    for lab in labels:
        for _ in range(80):
            s = int(rng.choice(patch_sizes))
            if s > n:
                continue
            y0 = int(rng.integers(0, n - s + 1)); x0 = int(rng.integers(0, n - s + 1))
            region = mask[y0:y0 + s, x0:x0 + s]
            if np.all(region == 0):
                region[:] = int(lab)          # the original drew a new random label here
                break
    return mask, {"kind": "patches", "n_patches": k}


SCENE_FNS = (satellite_scene, shapes_scene, patches_scene)


def random_scene(cfg: SimConfig, rng: np.random.Generator, image_size=None, library=None):
    """Draw a scene type according to `cfg.scene_probs` and build it."""
    idx = int(rng.choice(3, p=cfg.scene_probs))
    return SCENE_FNS[idx](cfg, rng, image_size=image_size, library=library)


# ======================================================================
# Stage 2 — spectra to cube (the 2a blend happens in generate_sample).
# ======================================================================

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

    Each pixel gets its material's whole spectrum copied along the wavelength
    axis, so before blurring every band has the same spatial pattern; only the
    value inside each region changes from band to band.  Background stays 0.
    """
    h, w = mask.shape
    cube = np.zeros((n_bands, h, w), float)
    for mid, spec in spectra.items():
        sel = mask == mid
        if sel.any():
            cube[:, sel] = spec[:, None]

    return cube


# ======================================================================
# Stage 3 — degradation.
# ======================================================================

def blur_cube(cube: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Stage 3a — per-band Gaussian blur, sigma growing with wavelength.

    This is the ONLY mechanism in the whole simulator that couples the spectral
    and spatial axes.  Everything else treats them independently, so a network
    cannot learn any spectral-spatial interaction beyond "redder is blurrier".

    Uses cv2 with BORDER_REFLECT_101, which matches scipy's mode="mirror".
    """
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


# ======================================================================
# Running the stages.
# ======================================================================

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
                    scene: str = "random", image_size: int | None = None) -> Sample:
    """One sample, with every intermediate retained.

    `scene` is "random" (default: drawn from cfg.scene_probs, the original's
    30/30/40 mix) | "satellite" | "shapes" | "patches".
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
