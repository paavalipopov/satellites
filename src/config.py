"""Configuration for the synthetic HSI generator.

Every number that the original pipeline hard-coded lives here, expressed in
physical terms rather than as pre-multiplied constants.

Defaults reproduce the regime of the *submitted manuscript* (350-1000 nm,
36 bands, sensor noise 0.02), which is the configuration whose reported
numbers we were able to reproduce.
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
    # The original draws ONE w per scene, shared by every material in it (== False below).
    blend_weight_per_material: bool = False

    # ---------------- degradation ----------------
    # (a) spectral noise, one draw per material, shared by all its pixels:
    #     s_noisy = s + N(0, (material_noise_scale * std(s))^2)
    material_noise_scale: float = 0.5

    # (b) diffraction PSF.  sigma_px(lambda) = (psf_k * lambda / D) / plate_scale
    aperture_m: float = 3.6 #TODO briefly explain how these vars relate to difraction PSF, do they affect the formula above, or are they used in psf_sigma_px?
    psf_k: float = 0.25
    pixel_pitch_m: float = 5e-6
    focal_length_m: float = 120.0

    # (c) additive per-pixel sensor noise, applied AFTER the blur
    sensor_noise_std: float = 0.02

    # ---------------- data hygiene ----------------
    # The measured spectra carry detector artifacts at the edges of each scan
    # (values from -9.8 to +15.6 at 350 nm). 
    # Detection and handling are separate here: the library ALWAYS
    # checks its resampled spectra against [0, 1] over the target grid and
    # reports what it finds; `on_invalid` decides what happens with that report
    # and `clip_reflectance` decides what happens to the values.
    on_invalid: str = "warn"               # "warn" | "raise" | "ignore"
    # Note: clipping FABRICATES a value (a hard 1.0, or 0.0) where there is no
    # valid measurement.  If the offending bands sit at the edge of your grid,
    # trimming the grid is more honest than clipping. 
    # The warning names the affected bands so you can do that.
    clip_reflectance: bool = True

    # Use only materials that actually have a measured irradiated trace.
    # 11 of the 46 sheets have none, and for those `blend()` returns pristine
    # and IGNORES w — so in a scene drawn at w=0.9 the Titanium antenna is still
    # pristine while everything around it is fully degraded.  Since w is shared
    # across the scene, that inconsistency is visible within a single sample.
    # Material ids are NOT renumbered, so a 47-class checkpoint still runs —
    # the excluded classes simply never appear.
    require_irradiated: bool = False # TODO I removed betacloth mention from the excluded list, make sure the code ignores it too, and briefly explain to me what was the problem with this material

    # On unmeasured spectra, the original behaviour past the last measured wavelength 
    # was `right=0.0`, which makes a material identifiable by where its measurements stop.
    # extrapolate_red controls what happens to the red end of the measured spectrum.
    extrapolate_red: str = "hold"          # "hold" | "zero" # TODO it seems to me that in addition to this handling we need to raise a warning like with on_invalid, maybe call it on_unmeasured_spectra

    # ---------------- geometry and POV ----------------
    # READ THIS BEFORE TRUSTING THE SATELLITE MASKS.
    #
    # At image_size=15 the spacecraft is grossly under-sampled.  With a 60 deg
    # FOV the focal length is f = 0.866 * image_size px, and at camera_height
    # 17 that makes:
    #     bus cylinder  (r=0.8)  ~1.2 px across
    #     antenna       (r=0.4)  ~0.6 px across
    #     connectors    (0.3 thick)  ~0.23 px across
    #     solar panels  (4 x 2)  ~3.1 x 1.5 px each, separated by ~9 px
    # The bus, antenna and connectors are sub-pixel and appear in the original's masks 
    # ONLY because `rasterizer="cv2"` fill degenerate slivers generously. 
    # 
    # See `SimConfig.resolved()` below if you want to avoid rasterization.

    panel_thickness: float = 0.01 # TODO briefly explain what this var controls and where 0.01 comes from
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
    # TODO let's remove these switches and the code that uses them, should make things simpler and easier to read
    legacy_drop_edge_faces: bool = False   # original skipped faces touching the frame edge
    legacy_flat_face_depth: bool = False   # original used one z per triangle
    legacy_triangle_bug: bool = False      # original's "triangle" had 2 points -> a line
    legacy_patch_label_bug: bool = False   # original discarded its own unique patch labels

    # ---------------- paths ----------------
    material_dir: str = "assets/materials"     # relative to root; originals in original_files/

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
        # TODO I think the vars used here introeuced above, remind their meaning here too (psf_k, aperture...)
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
    # TODO question about these methoids: they are resetting default config if I runt them, right? the default is defined above.
    # I think we can remove first two after all the other changes.
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
