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
    # explicit band centres (nm, ascending) replace the linspace grid above when set;
    # n_bands must match their count (e.g. a real sensor's bands)
    wavelengths: tuple[float, ...] | None = None

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

    # (b) diffraction PSF (see `psf_sigma_px`): sigma_px = (psf_k * lambda / D) / (p / f)
    aperture_m: float = 3.6          # D, telescope aperture; sets the scale lambda/D
    psf_k: float = 0.25              # Gaussian sigma in units of lambda/D (Airy fit ~0.42)
    pixel_pitch_m: float = 5e-6      # p, detector pixel size
    focal_length_m: float = 120.0    # f; p/f is the angle one pixel covers

    # (c) additive per-pixel sensor noise, applied AFTER the blur
    sensor_noise_std: float = 0.02

    # (d) overall brightness (off by default). When set, every scene is multiplied by ONE
    #     random factor 10 ** U(lo, hi), so the model sees objects of very different brightness.
    #     The factor comes from its own generator, so the scene and its noise are exactly what they
    #     would be without it (a scene with the gain off is the same scene).
    #       gain_stage "before_noise": the object is dimmer against the SAME sensor noise
    #                                  (a faint body: low signal-to-noise)
    #       gain_stage "after_noise":  the whole image, noise included, is scaled
    #                                  (a change of units: the signal-to-noise ratio is untouched)
    gain_log10_range: tuple[float, float] | None = None
    gain_stage: str = "before_noise"

    # (e) wavelength-dependent response (off by default). The real signal is sunlight x reflectance x
    #     atmosphere x optics x detector, and everything but the reflectance is a smooth curve over
    #     wavelength that the model has never seen. When response_log10_std > 0, every scene is
    #     multiplied by ONE random smooth curve over the bands (before the sensor noise, like the gain):
    #     log10(curve) interpolates `response_knots` random values with this sd, and has mean 0 over
    #     the bands, so it changes the spectral shape and leaves the overall brightness to the gain.
    response_log10_std: float = 0.0
    response_knots: int = 4              # 2 = a random tilt; more = more wiggles

    # (f) empty scenes (off by default). With this probability a scene holds NO object: the label is
    #     all background and the cube is only noise (and artifacts, below). The noise of such a scene
    #     is sensor_noise_std x 10**U(lo, hi), so it runs from nearly nothing (the exactly-zero
    #     background of estimate.fits) up to the full sensor noise.
    empty_prob: float = 0.0
    empty_noise_log10_range: tuple[float, float] = (-3.0, 0.0)

    # (g) halo (off by default): a faint wide skirt around the object, labelled BACKGROUND. It is the
    #     object blurred by a Gaussian of sigma U(halo_sigma_px) pixels, scaled so its peak is
    #     10**U(lo, hi) times the object's peak (estimate.fits has such pixels at 1e-4 to 1e-2 of the
    #     satellite). It is added before the response and gain, so it keeps the object's spectrum.
    halo_log10_range: tuple[float, float] | None = None
    halo_sigma_px: tuple[float, float] = (2.0, 6.0)

    # (h) restoration-like artifacts (off by default): noise that is not white Gaussian. A field that is
    #     smooth in space (sigma U(artifact_sigma_px) pixels), correlated between neighbouring bands
    #     (artifact_band_corr) and skewed to the high side (artifact_skew; 0 = symmetric), with an sd of
    #     10**U(log10 lo, log10 hi) in the same units as sensor_noise_std. Added after the sensor noise.
    artifact_std_range: tuple[float, float] | None = None
    artifact_sigma_px: tuple[float, float] = (0.7, 2.5)
    artifact_band_corr: float = 0.6
    artifact_skew: float = 0.5

    # (i) real artifacts (off by default): a .npy file of noise patches [bands, H, W] or
    #     [N, bands, H, W], in the units of the training cubes (see scripts/make_noise_bank.py).
    #     A random crop (random flip/rotation) times 10**U(noise_bank_log10_scale) is added after the
    #     sensor noise, in addition to the artifacts of (h).
    noise_bank: str | None = None
    noise_bank_log10_scale: tuple[float, float] = (0.0, 0.0)

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
    # The warning suggests how far to trim.
    clip_reflectance: bool = True

    # Use only materials that actually have a measured irradiated trace.
    # 11 of the 46 sheets have none, and for those `blend()` returns pristine
    # and IGNORES w — so in a scene drawn at w=0.9 the Titanium antenna is still
    # pristine while everything around it is fully degraded.  Since w is shared
    # across the scene, that inconsistency is visible within a single sample.
    # Material ids are NOT renumbered, so a 47-class checkpoint still runs —
    # the excluded classes simply never appear.
    require_irradiated: bool = False

    # Grid bands outside a sheet's measured range are filled, not measured: the blue
    # end always repeats the first value; the red end follows extrapolate_red.
    extrapolate_red: str = "hold"          # "hold": repeat the last value | "zero": set to 0
    on_unmeasured: str = "warn"            # "warn" | "raise" | "ignore"

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
    # ONLY because `rasterizer="cv2"` fills degenerate slivers generously. 
    # 
    # See `SimConfig.resolved()` below for a properly sampled render (60 px, exact rasterizer).

    panel_thickness: float = 0.01          # solar-panel thickness, scene units (original's)
    rasterizer: str = "cv2"                # "cv2" (as original) | "exact"

    # ---------------- scene mixture ----------------
    # Probability of each scene type, in the original's order.
    p_satellite: float = 0.30
    p_shapes: float = 0.30
    p_patches: float = 0.40

    # ---------------- paths ----------------
    material_dir: str = "assets/materials"     # relative to root; originals in original_files/

    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.gain_log10_range is not None:
            lo, hi = (float(v) for v in self.gain_log10_range)  # lists from configs too
            if lo > hi:
                raise ValueError(f"gain_log10_range={self.gain_log10_range}: lo must not exceed hi")
            object.__setattr__(self, "gain_log10_range", (lo, hi))
        if self.gain_stage not in ("before_noise", "after_noise"):
            raise ValueError(f"gain_stage={self.gain_stage!r}: 'before_noise' or 'after_noise'")
        for name, optional in (("halo_log10_range", True), ("artifact_std_range", True),
                               ("empty_noise_log10_range", False), ("halo_sigma_px", False),
                               ("artifact_sigma_px", False), ("noise_bank_log10_scale", False)):
            value = getattr(self, name)
            if value is None:
                if not optional:
                    raise ValueError(f"{name} must be a (lo, hi) pair")
                continue
            lo, hi = (float(v) for v in value)            # lists from configs too
            if lo > hi:
                raise ValueError(f"{name}={value}: lo must not exceed hi")
            object.__setattr__(self, name, (lo, hi))
        if self.artifact_std_range is not None and self.artifact_std_range[0] <= 0:
            raise ValueError("artifact_std_range must be positive (it is drawn log-uniformly)")
        if self.halo_sigma_px[0] <= 0 or self.artifact_sigma_px[0] <= 0:
            raise ValueError("halo_sigma_px and artifact_sigma_px must be positive")
        if not 0.0 <= self.empty_prob <= 1.0:
            raise ValueError(f"empty_prob={self.empty_prob}: a probability")
        if self.response_log10_std < 0 or self.response_knots < 2:
            raise ValueError("response_log10_std must be >= 0 and response_knots >= 2")
        if not 0.0 <= self.artifact_band_corr < 1.0:
            raise ValueError("artifact_band_corr must be in [0, 1)")
        if self.artifact_skew < 0:
            raise ValueError("artifact_skew must be >= 0")
        if self.wavelengths is None:
            return
        wavelengths = tuple(float(w) for w in self.wavelengths)  # lists from configs too
        object.__setattr__(self, "wavelengths", wavelengths)
        if len(wavelengths) != self.n_bands:
            raise ValueError(f"n_bands={self.n_bands}, but {len(wavelengths)} wavelengths given")
        if any(b <= a for a, b in zip(wavelengths, wavelengths[1:])):
            raise ValueError("wavelengths must be ascending; flip a red-to-blue cube instead")

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """The band centres.  This is the array to quote to collaborators."""
        if self.wavelengths is not None:
            return np.array(self.wavelengths)
        return np.linspace(self.lambda_min_nm, self.lambda_max_nm, self.n_bands)

    @property
    def psf_sigma_px(self) -> np.ndarray:
        """Per-band Gaussian PSF width, in pixels.

        theta = psf_k * lambda / D       blur angle in radians, D = aperture_m
        plate_scale = p / f              angle per pixel, p = pixel_pitch_m, f = focal_length_m
        sigma_px = theta / plate_scale   psf_k = Gaussian sigma in units of lambda/D
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
        fields = asdict(self)
        if fields["gain_log10_range"] is None:  # gain off: the fingerprint it had before the option existed
            del fields["gain_log10_range"], fields["gain_stage"]
        for key in _LATER_OPTIONS:               # same for every later option left at its default
            if fields[key] == _DEFAULTS[key]:
                del fields[key]
        blob = json.dumps(fields, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    # ------------------------------------------------------------------
    @classmethod
    def resolved(cls, image_size: int = 60, **kw) -> "SimConfig":
        """Same scene, properly sampled.  At image_size=60 the object spans
        ~49 px and the bus is ~2.4 px across.  Material accuracy measured
        higher here (0.89 vs 0.81), but the trained models never saw objects
        this large — it is extrapolation, not their training regime."""
        return cls(image_size=image_size, rasterizer="exact", **kw)

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)


# The options added after the first release; each is dropped from the fingerprint while at its default.
_LATER_OPTIONS = ("response_log10_std", "response_knots", "empty_prob", "empty_noise_log10_range",
                  "halo_log10_range", "halo_sigma_px", "artifact_std_range", "artifact_sigma_px",
                  "artifact_band_corr", "artifact_skew", "noise_bank", "noise_bank_log10_scale")
_DEFAULTS = {k: getattr(SimConfig(), k) for k in _LATER_OPTIONS}
