"""Configuration for the synthetic HSI generator.

Every number that the original pipeline hard-coded lives here, expressed in
physical terms rather than as pre-multiplied constants.

Defaults reproduce the regime of the *submitted manuscript* (350-1000 nm,
36 bands, sensor noise 0.02), which is the configuration whose reported
numbers we were able to reproduce.  The later June-2026 configuration used
`lambda_max_nm=2000` and `sensor_noise_std=0.2`; see `SimConfig.june_2026()`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field, replace

import numpy as np


@dataclass(frozen=True)
class SimConfig:
    # ---------------- spectral axis ----------------
    lambda_min_nm: float = 350.0
    lambda_max_nm: float = 1000.0
    n_bands: int = 36

    # ---------------- scene / camera ----------------
    image_size: int = 15
    camera_height: float = 17.0      # object is pushed to z = +17 in camera frame
    fov_deg: float = 60.0

    # ---------------- irradiation blend ----------------
    # s = (1 - w) * pristine + w * irradiated,  w ~ U(0, 1)
    # The original draws ONE w per scene, shared by every material in it.
    blend_weight_per_material: bool = False

    # ---------------- degradation ----------------
    # (a) spectral noise, one draw per material, shared by all its pixels:
    #     s_noisy = s + N(0, (material_noise_scale * std(s))^2)
    material_noise_scale: float = 0.5

    # (b) diffraction PSF.  sigma_px(lambda) = (psf_k * lambda / D) / plate_scale
    aperture_m: float = 3.6
    psf_k: float = 0.25
    pixel_pitch_m: float = 5e-6
    focal_length_m: float = 120.0

    # (c) additive per-pixel sensor noise, applied AFTER the blur
    sensor_noise_std: float = 0.02

    # ---------------- data hygiene ----------------
    # The measured spectra carry detector artifacts at the edges of each scan
    # (values from -9.8 to +15.6 at 350 nm).  The original propagated them
    # silently.  Detection and handling are separate here: the library ALWAYS
    # checks its resampled spectra against [0, 1] over the target grid and
    # reports what it finds; `on_invalid` decides what happens with that report
    # and `clip_reflectance` decides what happens to the values.
    #
    # Note clipping FABRICATES a value (a hard 1.0, or 0.0) where there is no
    # valid measurement.  If the offending bands sit at the edge of your grid —
    # which is the usual case — trimming the grid is honest where clipping is
    # not.  The warning names the affected bands so you can do that.
    on_invalid: str = "warn"               # "warn" | "raise" | "ignore"

    # Use only materials that actually have a measured irradiated trace.
    # 11 of the 46 sheets have none, and for those `blend()` returns pristine
    # and IGNORES w — so in a scene drawn at w=0.9 the Titanium antenna is still
    # pristine while everything around it is fully degraded.  Since w is shared
    # across the scene, that inconsistency is visible within a single sample.
    # Dropping them also removes both suspect foils (see SUSPECT_SHEETS) and
    # BetaCloth, whose 999 nm measurement cutoff gave it a unique zero tail.
    # Material ids are NOT renumbered, so a 47-class checkpoint still runs —
    # the excluded classes simply never appear.
    # Cost: the problem gets slightly HARDER, not easier (0.888 -> 0.860
    # per-pixel at 350-1000 nm), because several of the dropped materials were
    # the easy, highly distinctive ones.  You buy validity, not accuracy.
    require_irradiated: bool = False
    clip_reflectance: bool = True
    # Original behaviour past the last measured wavelength was `right=0.0`,
    # which makes a material identifiable by where its measurements stop.
    extrapolate_red: str = "hold"          # "hold" | "zero"

    # ---------------- geometry ----------------
    # READ THIS BEFORE TRUSTING THE SATELLITE MASKS.
    #
    # At image_size=15 the spacecraft is grossly under-sampled.  With a 60 deg
    # FOV the focal length is f = 0.866 * image_size px, and at camera_height
    # 17 that makes:
    #     bus cylinder  (r=0.8)  ~1.2 px across
    #     antenna       (r=0.4)  ~0.6 px across
    #     connectors    (0.3 thick)  ~0.23 px across
    #     solar panels  (4 x 2)  ~3.1 x 1.5 px each, separated by ~9 px
    # Only the panels have real coverage.  The bus, antenna and connectors are
    # sub-pixel and appear in the original's masks ONLY because it truncated
    # projected vertices to int32 and let cv2.fillConvexPoly fill degenerate
    # slivers generously.  With an exact pixel-centre coverage test they vanish
    # and foreground drops from ~13.5% to ~5.6%.
    #
    # So `rasterizer="cv2"` is the default: it is the only setting that
    # reproduces the mask statistics the trained models expect.  But the
    # honest reading is that the "satellite" in the training set is largely a
    # rasterisation artifact — a ~12 px dumbbell of two panels with a couple of
    # stray pixels between them.  Note the object always spans ~81% of the
    # frame whatever image_size is, because the FOV is fixed and the geometry
    # just fits; to actually resolve it you must raise image_size, not move the
    # camera.  See `SimConfig.resolved()`.
    panel_thickness: float = 0.01
    rasterizer: str = "cv2"                # "cv2" (as original) | "exact"

    # ---------------- scene mixture ----------------
    # Probability of each scene type, in the original's order.
    p_satellite: float = 0.30
    p_shapes: float = 0.30
    p_patches: float = 0.40

    # ---------------- faithfulness switches ----------------
    # Each reproduces a defect in the original.  All default to False.
    # Measured: at the configuration actually used (image_size=15,
    # camera_height=17) this rule NEVER fires — 0.0% of triangles touch the
    # frame edge, because the object always sits ~81% of the way across a
    # centred frame.  So the original's training masks were NOT chopped; this
    # is latent dead code.  It starts to bite only if the camera comes closer:
    # 1.6% of faces dropped at camera_height=12, 20.5% at 9, 70% at 5.
    legacy_drop_edge_faces: bool = False   # original skipped faces touching the frame edge
    legacy_flat_face_depth: bool = False   # original used one z per triangle
    legacy_triangle_bug: bool = False      # original's "triangle" had 2 points -> a line
    legacy_patch_label_bug: bool = False   # original discarded its own unique patch labels

    # ---------------- paths ----------------
    material_dir: str = "data/material_refined"

    # ------------------------------------------------------------------
    @property
    def wavelengths_nm(self) -> np.ndarray:
        """The band centres.  This is the array to quote to collaborators."""
        return np.linspace(self.lambda_min_nm, self.lambda_max_nm, self.n_bands)

    @property
    def psf_sigma_px(self) -> np.ndarray:
        """Per-band Gaussian PSF width, in pixels.

        theta = psf_k * lambda / D            (diffraction angle, radians)
        plate_scale = pixel_pitch / focal     (radians per pixel)
        sigma_px = theta / plate_scale

        NOTE the original computed this from a hard-coded 350-1000 nm ramp
        regardless of the actual grid, so its June `lambda_max_nm=2000` runs
        had blur that did not match their wavelengths.  Here it always
        follows `wavelengths_nm`.
        """
        theta = self.psf_k * (self.wavelengths_nm * 1e-9) / self.aperture_m
        plate_scale = self.pixel_pitch_m / self.focal_length_m
        return theta / plate_scale

    @property
    def scene_probs(self) -> np.ndarray:
        p = np.array([self.p_satellite, self.p_shapes, self.p_patches], float)
        return p / p.sum()

    def fingerprint(self) -> str:
        """Short stable hash of the whole config — stamp this into any sample
        you keep, so provenance is never lost the way it was last time."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    # ------------------------------------------------------------------
    @classmethod
    def june_2026(cls, **kw) -> "SimConfig":
        """The configuration the June 2026 SRR checkpoints were trained on."""
        return cls(lambda_max_nm=2000.0, sensor_noise_std=0.2, **kw)

    @classmethod
    def legacy(cls, **kw) -> "SimConfig":
        """Bug-for-bug faithful to the original `moduler_gen.py`."""
        return cls(
            clip_reflectance=False,
            on_invalid="ignore",
            extrapolate_red="zero",
            legacy_drop_edge_faces=True,
            legacy_flat_face_depth=True,
            legacy_triangle_bug=True,
            legacy_patch_label_bug=True,
            **kw,
        )

    @classmethod
    def resolved(cls, image_size: int = 60, **kw) -> "SimConfig":
        """Same scene, properly sampled.  At image_size=60 the object spans
        ~49 px, the bus is ~2.4 px across, and the exact rasteriser agrees with
        the cv2 one.  Material accuracy measured higher here (0.89 vs 0.81),
        but the trained models never saw objects this large — it is
        extrapolation, not their training regime."""
        return cls(image_size=image_size, rasterizer="exact", **kw)

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)
