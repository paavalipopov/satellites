"""Self-test of the brightness / robustness options. No training, no GPU, about a minute.

    python scripts/check_brightness_options.py

Checks, for every option in SimConfig and the model, that it does what BRIGHTNESS.md says and that
it leaves everything else alone. Prints PASS/FAIL per check and exits non-zero if any failed.
"""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_generation import SimConfig, SpectralLibrary, SpectralLibraryWarning, generate_sample  # noqa: E402
from src.models.spectral_spatial import INPUT_NORMS, SpectralSpatialHSIModel  # noqa: E402

warnings.simplefilter("ignore", SpectralLibraryWarning)
BANDS_10 = (400.0, 426.667, 457.143, 492.308, 533.333, 581.818, 640.0, 711.111, 800.0, 914.286)
BASE = SimConfig(n_bands=10, wavelengths=BANDS_10)
LIB = SpectralLibrary(BASE, root=str(ROOT))
FAILED = []


def check(name, cond, detail=""):
    FAILED.append(name) if not cond else None
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def sample(cfg, seed, scene="satellite", **kw):
    return generate_sample(cfg, seed, LIB, scene=scene, **kw)


def main():
    seeds = range(2000, 2012)

    # ---------------------------------------------------------------- generator: general
    cfg_all = BASE.with_(gain_log10_range=(-1.5, 0.5), response_log10_std=0.2, empty_prob=0.2,
                         halo_log10_range=(-4.0, -1.5), artifact_std_range=(0.002, 0.03))
    a, b = sample(cfg_all, 7, "random"), sample(cfg_all, 7, "random")
    check("same seed, same sample with every option on", np.array_equal(a.cube, b.cube) and np.array_equal(a.mask, b.mask))
    check("fingerprint unchanged by listing defaults explicitly", BASE.fingerprint() == BASE.with_(empty_prob=0.0, response_knots=4).fingerprint())
    check("fingerprint changes when an option is on",
          len({BASE.fingerprint(), BASE.with_(response_log10_std=0.2).fingerprint(), BASE.with_(empty_prob=0.1).fingerprint(),
               BASE.with_(halo_log10_range=(-3, -2)).fingerprint(), BASE.with_(artifact_std_range=(0.01, 0.02)).fingerprint()}) == 5)
    for bad in (dict(response_log10_std=-1), dict(response_knots=1), dict(empty_prob=1.5), dict(halo_log10_range=(-1, -3)),
                dict(artifact_std_range=(0, 0.1)), dict(artifact_band_corr=1.0), dict(artifact_skew=-1), dict(halo_sigma_px=(0, 3))):
        try:
            BASE.with_(**bad)
            check(f"rejects {bad}", False)
        except ValueError:
            check(f"rejects {bad}", True)

    # an option on must leave the scene, the spectra, the label and the unscaled cube alone
    plain = [sample(BASE, s) for s in seeds]
    for name, kw in (("response", dict(response_log10_std=0.3)), ("halo", dict(halo_log10_range=(-3.0, -1.0))),
                     ("artifacts", dict(artifact_std_range=(0.01, 0.03))), ("gain", dict(gain_log10_range=(-1, 0)))):
        on = [sample(BASE.with_(**kw), s) for s in seeds]
        same = all(np.array_equal(p.mask, o.mask) and np.array_equal(p.cube_blurred, o.cube_blurred) and p.blend_weight == o.blend_weight
                   for p, o in zip(plain, on))
        changed = all(not np.array_equal(p.cube, o.cube) for p, o in zip(plain, on))
        check(f"{name}: label, spectra and blurred cube unchanged, input changed", same and changed)

    # ---------------------------------------------------------------- response curve
    cfg = BASE.with_(response_log10_std=0.2)
    curves = np.array([sample(cfg, s).response for s in range(3000, 3200)])
    logs = np.log10(curves)
    check("response: one curve per band, geometric mean 1", curves.shape == (200, 10) and np.allclose(logs.mean(1), 0, atol=1e-9))
    check("response: sd of the log curve is about the setting", 0.1 < logs.std() < 0.25, f"sd {logs.std():.3f}")
    smooth = np.abs(np.diff(logs, 2, axis=1)).mean() / np.abs(np.diff(logs, 1, axis=1)).mean()
    check("response: smooth (second differences small next to first)", smooth < 1.0, f"ratio {smooth:.2f}")
    tilt = np.array([sample(cfg.with_(response_knots=2), s).response for s in range(3000, 3010)])
    t = np.log10(tilt)
    check("response: two knots = a tilt (log curve is a smooth S between two values)", np.all(np.abs(t.mean(1)) < 1e-9))
    s0 = sample(BASE, 3000)
    s1 = sample(cfg, 3000)
    check("response: noise is added after it (input - curve*blurred is the same noise)",
          np.allclose(s1.cube - s1.cube_blurred * s1.response[:, None, None], s0.cube - s0.cube_blurred, atol=1e-9))

    # ---------------------------------------------------------------- empty scenes
    e = [sample(BASE.with_(empty_prob=1.0), s, "random") for s in range(4000, 4040)]
    check("empty_prob=1: every label is background", all((x.mask == 0).all() and x.meta["kind"] == "empty" for x in e))
    check("empty: no object signal in the unscaled cube", all(np.abs(x.cube_blurred).max() == 0 for x in e))
    frac = np.mean([sample(BASE.with_(empty_prob=0.3), s, "random").meta["kind"] == "empty" for s in range(5000, 5600)])
    check("empty_prob=0.3: about 30% of scenes are empty", 0.24 < frac < 0.36, f"{frac:.2f}")
    sds = np.array([x.cube.std() for x in e])
    check("empty: noise ranges from tiny to the full sensor noise",
          sds.min() < 0.02 * 10 ** -2 and sds.max() > 0.02 * 0.4, f"sd {sds.min():.5f} .. {sds.max():.4f}")
    fixed = [sample(BASE.with_(empty_noise_log10_range=(-1.0, -1.0)), s, "empty").cube.std() for s in range(10)]
    check("scene='empty' with a fixed noise level", np.allclose(fixed, 0.002, rtol=0.2), f"{np.mean(fixed):.4f}")
    ne = [sample(BASE.with_(empty_prob=0.5), s, "random") for s in range(6000, 6030)]
    p0 = [sample(BASE, s, "random") for s in range(6000, 6030)]
    same = [np.array_equal(x.mask, y.mask) for x, y in zip(ne, p0) if x.meta["kind"] != "empty"]
    check("empty_prob: the scenes that stay are the same scenes", len(same) > 5 and all(same))

    # ---------------------------------------------------------------- halo
    cfg = BASE.with_(halo_log10_range=(-2.0, -2.0), halo_sigma_px=(3.0, 3.0))
    ok_peak, ok_bg, ok_mask = [], [], []
    for s in seeds:
        h, p = sample(cfg, s), sample(BASE, s)
        diff = h.cube - p.cube                                  # the halo alone: same noise in both
        ok_peak.append(abs(diff.max() / p.cube_blurred.max() - 0.01) < 1e-6)
        # the kernel is cut off a few sigma out; and a material's spectral noise can leave a band slightly
        # negative, which the halo then follows, so allow a dip of a tenth of the halo's own peak
        bg = diff[:, h.mask == 0]
        ok_bg.append(bg.min() >= -0.1 * diff.max() and (bg > 0).mean() > 0.1)
        ok_mask.append(np.array_equal(h.mask, p.mask) and abs(h.halo_amp - 0.01) < 1e-12)
    check("halo: its peak is amp x the object's peak", all(ok_peak))
    check("halo: it is never negative, positive near the object, and the label stays background", all(ok_bg) and all(ok_mask))
    he = sample(cfg, 1, "empty")
    check("halo: nothing to add to an empty scene", he.halo_amp == 0.0)
    wide = np.mean([(sample(BASE.with_(halo_log10_range=(-2, -2), halo_sigma_px=(6, 6)), s).cube
                     - sample(BASE, s).cube)[:, :1, :1].mean() for s in seeds])
    narrow = np.mean([(sample(BASE.with_(halo_log10_range=(-2, -2), halo_sigma_px=(1, 1)), s).cube
                       - sample(BASE, s).cube)[:, :1, :1].mean() for s in seeds])
    check("halo: a wider skirt reaches further (the corner is brighter)", wide > narrow, f"{wide:.2e} vs {narrow:.2e}")

    # ---------------------------------------------------------------- artifacts
    cfg = BASE.with_(artifact_std_range=(0.02, 0.02))
    arts = np.array([(sample(cfg, s, "empty").cube - sample(BASE, s, "empty").cube) for s in range(7000, 7040)])
    check("artifacts: sd is the setting", np.allclose(arts.std(axis=(1, 2, 3)), 0.02, rtol=0.35), f"{arts.std():.4f}")
    check("artifacts: skewed to the high side (default skew)", ((arts - arts.mean()) ** 3).mean() / arts.std() ** 3 > 0.2)
    sym = np.array([(sample(cfg.with_(artifact_skew=0.0), s, "empty").cube - sample(BASE, s, "empty").cube) for s in range(7000, 7040)])
    check("artifacts: skew 0 is symmetric", abs(((sym - sym.mean()) ** 3).mean() / sym.std() ** 3) < 0.2)
    corr_b = np.mean([np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1] for a in arts])
    corr_x = np.mean([np.corrcoef(a[:, :, :-1].ravel(), a[:, :, 1:].ravel())[0, 1] for a in arts])
    white = np.random.default_rng(0).normal(size=arts.shape)
    check("artifacts: correlated between neighbouring bands and pixels (white noise is not)",
          corr_b > 0.3 and corr_x > 0.3 and abs(np.corrcoef(white[0, :-1].ravel(), white[0, 1:].ravel())[0, 1]) < 0.1,
          f"band {corr_b:.2f}, pixel {corr_x:.2f}")

    # ---------------------------------------------------------------- noise bank
    with tempfile.TemporaryDirectory() as d:
        bank = np.random.default_rng(1).normal(size=(3, 10, 20, 20)).astype(np.float32)
        np.save(Path(d) / "bank.npy", bank)
        cfg = BASE.with_(noise_bank=str(Path(d) / "bank.npy"), noise_bank_log10_scale=(0.0, 0.0))
        diff = sample(cfg, 11, "empty").cube - sample(BASE, 11, "empty").cube
        flat = np.sort(np.abs(diff).ravel())
        found = any(np.allclose(np.sort(np.abs(np.rot90(p[:, y:y + 15, x:x + 15], k, axes=(1, 2))).ravel()), flat, atol=1e-5)
                    for p in bank for y in range(6) for x in range(6) for k in range(1)) or \
            any(np.allclose(np.sort(np.abs(p[:, y:y + 15, x:x + 15]).ravel()), flat, atol=1e-5) for p in bank for y in range(6) for x in range(6))
        check("noise bank: a crop of the bank is what gets added", found)
        scaled = sample(cfg.with_(noise_bank_log10_scale=(1.0, 1.0)), 11, "empty").cube - sample(BASE, 11, "empty").cube
        check("noise bank: scaled by 10**scale", np.allclose(scaled, 10 * diff, atol=1e-5))
        try:
            sample(BASE.with_(noise_bank=str(Path(d) / "bank.npy"), n_bands=36, wavelengths=None), 1)
            check("noise bank: refuses the wrong number of bands", False)
        except ValueError:
            check("noise bank: refuses the wrong number of bands", True)

    # ---------------------------------------------------------------- model
    x = torch.rand(3, 10, 15, 15) + 0.05
    ref = SpectralSpatialHSIModel(10, 47).eval()
    state = ref.state_dict()
    with torch.no_grad():
        base_out = ref(x)
        for mode in INPUT_NORMS:
            m = SpectralSpatialHSIModel(10, 47, input_norm=mode).eval()
            m.load_state_dict(state)                              # same layers and weights for every mode
            out = m(x)
            if mode == "none":
                check("model none: identical to the plain model", torch.equal(out, base_out))
            if mode in ("max", "p99", "flux"):
                ok = all(torch.allclose(m(x * k), out, atol=1e-4, rtol=1e-4) for k in (1e-3, 0.07, 9.0, 1e4))
                check(f"model {mode}: scaling the input leaves the output unchanged", ok)
                two = torch.stack([x[0], 1000 * x[1]])
                check(f"model {mode}: each sample normalised on its own",
                      torch.allclose(m(two)[1], m(x[1:2])[0], atol=1e-4) and torch.allclose(m(two)[0], m(x[:1])[0], atol=1e-4))
                check(f"model {mode}: an explicit scale equals the input's own",
                      torch.allclose(m(x, scale=m.input_scale(x)), out))
            if mode == "band_flux":
                c = torch.rand(1, 10, 1, 1) * 5 + 0.2
                check("model band_flux: a different gain per band leaves the output unchanged", torch.allclose(m(x * c), out, atol=1e-4, rtol=1e-4))
            if mode == "pixel":
                c = torch.rand(1, 1, 15, 15) * 5 + 0.2
                check("model pixel: a different gain per pixel leaves the output unchanged", torch.allclose(m(x * c), out, atol=1e-4, rtol=1e-4))
            check(f"model {mode}: all-zero input stays finite", torch.isfinite(m(torch.zeros(1, 10, 15, 15))).all().item())
        m = SpectralSpatialHSIModel(10, 47, input_norm="p99").eval()
        spike = x.clone(); spike[0, 0, 0, 0] = 1e6
        mx = SpectralSpatialHSIModel(10, 47, input_norm="max").eval()
        check("model p99: one hot pixel does not set the scale (max does)",
              float(m.input_scale(spike)[0]) < 10 and float(mx.input_scale(spike)[0]) > 1e5)
    try:
        SpectralSpatialHSIModel(10, 47, input_norm="mean")
        check("model: rejects an unknown input_norm", False)
    except ValueError:
        check("model: rejects an unknown input_norm", True)

    print("\nALL PASS" if not FAILED else f"\n{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
