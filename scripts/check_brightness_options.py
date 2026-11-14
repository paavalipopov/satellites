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


def check_fits_reading():
    """scripts/fits_reading.py: the switch between the old reading of estimate.fits and the collaborators' recipe."""
    import hashlib
    import inspect

    sys.path.insert(0, str(ROOT / "scripts"))
    import fits_reading as fr
    import radiometry

    raw = (ROOT / "scripts/radiometry.py").read_bytes()
    check("scripts/radiometry.py is the collaborators' file, unchanged",
          hashlib.md5(raw).hexdigest() == "184427b6a7622fd247df8bbcdbff1ab7", hashlib.md5(raw).hexdigest())
    lam = fr.wavelength_grid()                                  # red -> blue, as the FITS files
    check("slice wavelengths rebuilt as the table (914.29 ... 400 nm, 6400/n)",
          np.allclose(lam, 6400.0 / np.arange(7, 17), atol=1e-9) and np.allclose(lam[[0, 3, 9]], [914.2857, 640.0, 400.0], atol=1e-3))
    check("the model's band centres are the same wavelengths", np.allclose(sorted(lam), BANDS_10, atol=1e-3))
    blue_red = np.array(sorted(lam))
    try:
        import pvlib
        solar = pvlib.spectrum.get_reference_spectra(standard="ASTM G173-03")["extraterrestrial"]
        old = np.interp(blue_red, solar.index, solar.values) * blue_red ** 2 / 6400       # the expression FitsScene used before
        check("reading 'solar' is exactly the previous divisor E * dlam", np.array_equal(fr.divisor("solar", blue_red), old))
    except ImportError:
        print("SKIP  reading 'solar' (pvlib not installed)")
    import brightness_eval as ev
    check("FitsScene reads 'solar' unless told otherwise", inspect.signature(ev.FitsScene.__init__).parameters["reading"].default == "solar")
    for name in fr.READINGS[1:]:
        d = fr.divisor(name, blue_red)
        check(f"reading '{name}': positive and finite", bool(np.all(np.isfinite(d)) and np.all(d > 0)))
        check(f"reading '{name}': does not depend on the order the wavelengths are given in",
              np.allclose(fr.divisor(name, blue_red[::-1])[::-1], d, rtol=1e-12))
    check("mag only sets the overall constant (mag 1 is 100x mag 6)",
          np.allclose(fr.divisor("radiometry", blue_red, 1.0) / fr.divisor("radiometry", blue_red, 6.0), 100.0))
    sun = fr.divisor("radiometry_sun", blue_red) / fr.divisor("radiometry", blue_red)
    b = radiometry.blackbody_flambda(blue_red, fr.T_SUN)
    check("the sun version differs from the other by the 6000 K sun's shape only", np.allclose(sun / b, (sun / b)[0], rtol=1e-9))
    try:
        fr.divisor("nonsense", blue_red)
        check("rejects an unknown reading", False)
    except ValueError:
        check("rejects an unknown reading", True)

    est_path, obj_path = ROOT / "assets/new_input/estimate.fits", ROOT / "assets/new_input/object.fits"
    if not (est_path.is_file() and obj_path.is_file()):
        print("SKIP  checks on assets/new_input (no estimate.fits / object.fits)")
        return
    from astropy.io import fits
    est, obj = (fits.getdata(p).astype(float) for p in (est_path, obj_path))
    # (1) the numbers recover_reflectance.py printed on these two files (TRUTH_INCLUDES_SUN = True, mag 1; blue -> red)
    d = fr.divisor("radiometry_sun", lam, 1.0)
    re_, ro = fr.to_reflectance(est, d), fr.to_reflectance(obj, d)
    m = np.isfinite(re_).all(0) & np.isfinite(ro).all(0)
    o = np.argsort(lam)
    check("reproduces the collaborators' script: restored disk spectrum",
          np.allclose(fr.disk_spectrum(est, d, m)[o], [1.0, 0.7708, 0.5763, 0.4168, 0.3040, 0.2533, 0.2581, 0.3181, 0.4633, 0.8013], atol=1e-4))
    check("reproduces the collaborators' script: truth disk spectrum",
          np.allclose(fr.disk_spectrum(obj, d, m)[o], [1.0, 0.7698, 0.5724, 0.4179, 0.3137, 0.2643, 0.2712, 0.3407, 0.5086, 0.8965], atol=1e-4))
    # (2) which reading gives the materials of the reflectance figure? Group the truth pixels by spectrum:
    # the large group is the solar panel (figure: 68% at 400 nm, 3.5% at 640 nm); the 4-pixel group is aluminized mylar
    # (figure: flat, 58-70% from 400 to 914 nm).
    fg = obj.sum(0) > 0
    spectra = (obj[:, fg] / obj.sum(0)[fg]).round(6).T
    _, inv, counts = np.unique(spectra, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    panel = obj[:, fg][:, inv == counts.argmax()].mean(1)
    mylar = obj[:, fg][:, inv == np.where(counts == 4)[0][0]].mean(1)
    i400, i640 = int(np.argmin(lam)), int(np.argmin(abs(lam - 640)))
    r = {n: (panel / fr.divisor(n, lam), mylar / fr.divisor(n, lam)) for n in fr.READINGS}
    ratio = {n: v[0][i400] / v[0][i640] for n, v in r.items()}
    flat = {n: v[1].max() / v[1].min() for n, v in r.items()}
    check("'radiometry': solar panel 400 nm / 640 nm is about the figure's 19", 12 < ratio["radiometry"] < 30, f"{ratio['radiometry']:.1f}")
    check("'solar' (previous reading): the same ratio is far below the figure's", ratio["solar"] < 8, f"{ratio['solar']:.1f}")
    check("'radiometry': aluminized mylar is flat, as in the figure (max/min < 1.35)", flat["radiometry"] < 1.35, f"{flat['radiometry']:.2f}")
    check("'radiometry_sun' and 'solar' do not make mylar flat",
          flat["radiometry_sun"] > 1.8 and flat["solar"] > 3, f"{flat['radiometry_sun']:.2f}, {flat['solar']:.2f}")
    check("'radiometry': values are the size a model can use (brightest pixel 0.01 - 10)",
          0.01 < (obj / fr.divisor("radiometry", lam)[:, None, None]).max() < 10)


def check_presets():
    """scripts/brightness_presets.py: the three single-change variants are the parts of empty_max, and every preset is a valid config."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import brightness_presets as bp

    def sim(name):
        return bp.sim_options(name)

    gain_keys = {"gain_log10_range", "gain_stage"}
    parts = {n: set(sim(n)) - gain_keys for n in ("add_empty", "add_halo", "add_noise")}
    full = set(sim("empty_max")) - gain_keys
    check("add_empty / add_halo / add_noise: disjoint parts that make up empty_max",
          set().union(*parts.values()) == full and sum(len(v) for v in parts.values()) == len(full))
    check("add_*: every part keeps the brightness augmentation and the max normalisation",
          all(bp.model_options(n) == {"input_norm": "max"} and gain_keys <= set(sim(n)) for n in parts))
    check("empty_max, full_max, full_p99, full_raw still use all three parts",
          all(full <= set(sim(n)) for n in ("empty_max", "full_max", "full_p99", "full_raw")))
    prints = set()
    for name in bp.PRESETS:
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sim(name).items()}
        try:
            prints.add(BASE.with_(**kw).fingerprint())
        except Exception as e:                                     # noqa: BLE001
            check(f"preset {name} builds a valid SimConfig", False, str(e))
    check("every preset builds a valid SimConfig", True)
    distinct = len({repr(sorted(sim(n).items())) for n in bp.PRESETS})        # e.g. aug and aug_norm share their data
    check("presets with different data options give different configs, equal options the same (fingerprints)",
          len(prints) == distinct, f"{len(prints)} fingerprints, {distinct} distinct option sets")


def check_auto_scale():
    """scripts/auto_scale.py: scaling in front of the model (no model needed: a stand-in predictor records what it was given)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import auto_scale as au

    def cube_with(blobs, shape=(10, 40, 40), noise=0.0, seed=0):
        """blobs: [(y0, y1, x0, x1, brightness)]; each pixel's spectrum is flat times its brightness."""
        c = np.random.default_rng(seed).normal(0, noise, shape) if noise else np.zeros(shape)
        for y0, y1, x0, x1, b in blobs:
            c[:, y0:y1, x0:x1] = b
        return c

    two = cube_with([(5, 10, 5, 10, 1.0), (25, 30, 25, 30, 0.05)])
    cores, terr = au.object_blobs(two)
    check("auto_scale: two separate blobs are found as two objects", int(cores.max()) == 2 and (cores[7, 7], cores[27, 27]) == (1, 2))
    check("auto_scale: territories do not overlap and each holds its own object",
          bool(((terr == 1) & (cores == 2)).sum() == 0 and (terr[cores == 1] == 1).all() and (terr[cores == 2] == 2).all()))
    check("auto_scale: the skirt is only `margin` pixels wide", int((terr > 0).sum()) <= 2 * (5 + 2 * au.MARGIN) ** 2 and terr[0, 0] == 0)
    plans = {m: au.plan(two, m) for m in au.MODES}
    check("auto_scale: none = one run, scale 1", len(plans["none"]) == 1 and plans["none"][0][1] == 1.0)
    check("auto_scale: single = one run, scale = brightest object value", len(plans["single"]) == 1 and abs(plans["single"][0][1] - 1.0) < 1e-12)
    check("auto_scale: per_object = one run per object, each scaled by its own maximum",
          len(plans["per_object"]) == 2 and abs(plans["per_object"][0][1] - 1.0) < 1e-12 and abs(plans["per_object"][1][1] - 0.05) < 1e-12)
    check("auto_scale: single_mean = the mean of all object values; single_p99 = their 99th percentile",
          abs(plans["single_mean"][0][1] - 0.525) < 1e-9 and abs(plans["single_p99"][0][1] - 1.0) < 1e-9)
    check("auto_scale: per_object_mean / per_object_p99 = each object's own mean / 99th percentile",
          [round(sc, 9) for _, sc in plans["per_object_mean"]] == [1.0, 0.05] and [round(sc, 9) for _, sc in plans["per_object_p99"]] == [1.0, 0.05])

    seen = []

    def make(runs):
        def stand_in(x):                              # what a model would be given; answers "object 1" wherever the input is non-zero
            seen.append(x.numpy().copy())
            return np.repeat((x.sum(1) > 0).long().numpy()[None], runs, axis=0)       # [runs, 1, H, W]
        return stand_in

    out = au.predict_scaled(make(3), two, "per_object", n_runs=3)
    check("auto_scale: per_object: every object is given to the model with its own maximum = 1",
          len(seen) == 2 and all(abs(float(x.max()) - 1.0) < 1e-6 for x in seen), f"{[float(x.max()) for x in seen]}")
    check("auto_scale: per_object: an object's run does not contain the other object", float(seen[0][0, :, 25:30, 25:30].max()) == 0.0)
    def by_run(x):                                    # model run r answers material r+1 everywhere it is asked
        return np.broadcast_to(np.arange(1, 4)[:, None, None, None], (3, 1) + tuple(x.shape[-2:])).copy()

    out = au.predict_scaled(by_run, two, "per_object", n_runs=3)
    check("auto_scale: per_object: answers are put back in place, background elsewhere, one answer per model run",
          out.shape == (3, 40, 40) and all(bool((out[r][terr > 0] == r + 1).all()) for r in range(3))
          and bool((out[:, terr == 0] == 0).all()))
    seen.clear()
    au.predict_scaled(make(1), two, "single", n_runs=1)
    check("auto_scale: single: the whole image is divided by one number", len(seen) == 1 and abs(float(seen[0].max()) - 1.0) < 1e-6
          and abs(float(seen[0][0, 0, 27, 27]) - 0.05) < 1e-6)
    seen.clear()
    au.predict_scaled(make(1), two, "none", n_runs=1)
    check("auto_scale: none: the image is given as it is", len(seen) == 1 and np.allclose(seen[0][0], two))

    empty = [cube_with([], noise=0.02, seed=s_) for s_ in range(5)] + [np.zeros((10, 40, 40))]
    seen.clear()
    outs = [au.predict_scaled(make(2), c, m, n_runs=2) for c in empty for m in ("single", "per_object")]
    check("auto_scale: images with nothing in them (noise only, or all zero) are not given to the model: all background",
          len(seen) == 0 and all(o.max() == 0 for o in outs))
    noisy = cube_with([(10, 16, 10, 16, 1.0)], noise=0.02, seed=3)
    check("auto_scale: with noise, the object is found and the noise is not (one object)", int(au.object_blobs(noisy)[0].max()) == 1)
    check("auto_scale: a one-pixel speck is ignored", int(au.object_blobs(cube_with([(5, 10, 5, 10, 1.0), (30, 31, 30, 31, 1.0)]))[0].max()) == 1)
    check("auto_scale: touching objects are ONE object (stated limit)",
          int(au.object_blobs(cube_with([(5, 10, 5, 10, 1.0), (10, 15, 10, 15, 0.05)]))[0].max()) == 1)
    hot = cube_with([(5, 10, 5, 10, 1.0)]); hot[0, 7, 7] = 10.0
    check("auto_scale: q=0.99 resists one hot pixel inside an object, the default q=1 is the maximum",
          au.plan(hot, "single")[0][1] == 10.0 and au.plan(hot, "single", q=0.99)[0][1] < 2.0)
    # the second search for faint objects (the companion of estimate.fits: 1 px, about 0.1% of the brightest)
    def scene_with_companion(extra_speck=None):
        c = cube_with([(10, 26, 10, 13, 1.0)], shape=(10, 41, 41))   # a bright 16 x 3 satellite: band sum 10 per pixel
        c[:, 9, 9:14] = 0.004                                         # specks hugging it: 0.4% of the brightest
        c[:, 18, 30] = 0.001                                          # the companion: 0.1%, 17 px away
        c[:, 5, 35] = 0.0004                                          # a far speck: 0.04%
        if extra_speck is not None:
            c[:, extra_speck[0], extra_speck[1]] = 0.0008             # a far speck nearly as bright as the companion
        return c

    comp = scene_with_companion()
    plain, _ = au.object_blobs(comp)
    faint, _ = au.object_blobs(comp, faint=True)
    check("auto_scale faint: the first search alone misses the companion (below 1% of the brightest)", int(plain.max()) == 1 and plain[18, 30] == 0)
    check("auto_scale faint: the second search finds it, and not the specks hugging the satellite or the weaker far speck",
          int(faint.max()) == 2 and faint[18, 30] == 2 and faint[5, 35] == 0 and faint[9, 11] == 0)
    check("auto_scale faint: a candidate not clearly brighter than the next speck is not taken",
          int(au.object_blobs(scene_with_companion(extra_speck=(30, 5)), faint=True)[0].max()) == 1)
    lone = np.zeros((10, 41, 41)); lone[:, 20, 20] = 0.001
    check("auto_scale faint: no bright object, no second search (an empty window stays empty)", int(au.object_blobs(lone, faint=True)[0].max()) == 0)
    check("auto_scale faint: with noise and one object, no faint object is invented",
          int(au.object_blobs(cube_with([(10, 16, 10, 16, 1.0)], noise=0.02, seed=4), faint=True)[0].max()) == 1)
    pf = au.plan(comp, "per_object_faint")
    check("auto_scale per_object_faint: two runs, the companion scaled by its own brightest value",
          len(pf) == 2 and abs(pf[0][1] - 1.0) < 1e-12 and abs(pf[1][1] - 0.001) < 1e-12 and bool(pf[1][0][18, 30]))
    check("auto_scale per_object (no faint search) is unchanged: one run", len(au.plan(comp, "per_object")) == 1)
    try:
        au.plan(two, "median")
        check("auto_scale: rejects an unknown mode", False)
    except ValueError:
        check("auto_scale: rejects an unknown mode", True)


def check_pipeline_parts():
    """src/losses.py, the new variants of scripts/brightness_presets.py and scripts/postprocess.py."""
    import os

    import torch.nn.functional as F

    sys.path.insert(0, str(ROOT / "scripts"))
    import brightness_eval as ev
    import brightness_presets as bp
    import postprocess as pp
    from src.losses import ObjectAwareLoss, build_criterion

    # ---- losses
    check("loss: no section (or an empty one) is the plain torch.nn.CrossEntropyLoss, as before",
          type(build_criterion(None)) is torch.nn.CrossEntropyLoss and type(build_criterion({})) is torch.nn.CrossEntropyLoss
          and type(build_criterion({"fg_weight": 1.0, "dice": 0.0})) is torch.nn.CrossEntropyLoss)
    for bad in ({"fg_weigt": 3.0}, {"fg_weight": 0.5}, {"dice": -1.0}):
        try:
            build_criterion(bad)
            check(f"loss: rejects {bad}", False)
        except ValueError:
            check(f"loss: rejects {bad}", True)
    g = torch.Generator().manual_seed(0)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, 2:5, 2:5] = torch.randint(1, 4, (2, 3, 3), generator=g)
    logits = torch.randn(2, 5, 8, 8, generator=g)
    w = torch.tensor([1.0, 3.0, 3.0, 3.0, 3.0])
    check("loss: fg_weight=3 is the cross-entropy with weight 1 for background and 3 for every material",
          torch.allclose(ObjectAwareLoss(3.0)(logits, target), F.cross_entropy(logits, target, weight=w), atol=1e-6))
    drops = torch.full((2, 5, 8, 8), -8.0); drops[:, 0] = 8.0                      # a model that calls everything background
    check("loss: dropping the object costs more with fg_weight and with dice than in the plain loss",
          float(ObjectAwareLoss(3.0)(drops, target)) > float(F.cross_entropy(drops, target))
          and float(ObjectAwareLoss(1.0, 1.0)(drops, target)) - float(F.cross_entropy(drops, target)) > 0.9)
    right = torch.full((2, 5, 8, 8), -20.0)
    right.scatter_(1, target[:, None], 20.0)                                       # confidently correct
    check("loss: a perfect answer costs about nothing, also with dice", float(ObjectAwareLoss(3.0, 1.0)(right, target)) < 1e-3)
    lg = logits.clone().requires_grad_(True)
    ObjectAwareLoss(3.0, 0.5)(lg.bfloat16(), target).backward()
    check("loss: takes bfloat16 logits and gives gradients", lg.grad is not None and bool(torch.isfinite(lg.grad).all()))

    # ---- presets
    names = ("fg3", "fg10", "fg3_dice", "lightnoise", "bank")
    check("presets: fg3, fg10, fg3_dice, lightnoise and bank exist", all(n in bp.PRESETS for n in names))
    check("presets: loss variants give +loss.* and nothing else new (original data, no scaling inside)",
          all(bp.model_options(n) == {} and bp.sim_options(n) == {} and bp.data_options(n) == {} for n in ("fg3", "fg10", "fg3_dice"))
          and bp.hydra_overrides("fg3_dice") == ["+loss.fg_weight=3.0", "+loss.dice=0.5"]
          and bp.hydra_overrides("fg10") == ["+loss.fg_weight=10.0"])
    check("presets: no variant of the earlier rounds has a loss section",
          all(bp.loss_options(n) == {} for n in bp.PRESETS if n not in names + ("original_retrained",) + tuple(bp.STAGE3)))
    check("presets: original_retrained is the original exactly: plain cross-entropy, no data / model / generator change",
          type(build_criterion(bp.loss_options("original_retrained"))) is torch.nn.CrossEntropyLoss
          and bp.sim_options("original_retrained") == {} and bp.data_options("original_retrained") == {}
          and bp.model_options("original_retrained") == {} and bp.hydra_overrides("original_retrained") == ["+loss.fg_weight=1.0"])
    check("stages: both stages compare against original_retrained",
          all("original_retrained" in bp.GROUPS[g] for g in ("stage1_compare", "stage2_compare")))
    check("presets: lightnoise is weaker restoration-like noise than empty_max's",
          bp.sim_options("lightnoise")["artifact_std_range"][1] < bp.sim_options("empty_max")["artifact_std_range"][1])
    old = os.environ.pop("NOISE_BANK", None)
    try:
        bp.hydra_overrides("bank")
        check("presets: bank without $NOISE_BANK stops with a message", False)
    except SystemExit:
        check("presets: bank without $NOISE_BANK stops with a message", True)
    os.environ["NOISE_BANK"] = "/some/bank.npy"
    check("presets: bank with $NOISE_BANK adds +data.params.sim.noise_bank", "+data.params.sim.noise_bank=/some/bank.npy" in bp.hydra_overrides("bank"))
    os.environ.pop("NOISE_BANK")
    if old is not None:
        os.environ["NOISE_BANK"] = old

    # ---- the two stages
    G = bp.GROUPS
    check("stages: every name in every group is a variant", all(n in bp.PRESETS for g in G.values() for n in g))
    check("stages: each stage compares the variants it trains", set(G["stage1_train"]) <= set(G["stage1_compare"])
          and set(G["stage2_train"]) <= set(G["stage2_compare"]))
    check("stage 1 changes ONLY the brightness handling: no halo, no extra noise, no loss change in its new variants",
          all(not ({"halo_log10_range", "artifact_std_range", "empty_prob", "response_log10_std"} & set(bp.sim_options(n)))
              and bp.loss_options(n) == {} for n in G["stage1_train"]))
    check("stage 2 keeps the brightness handling of the original: no random brightness, no pairs, no scaling inside",
          all("gain_log10_range" not in bp.sim_options(n) and bp.data_options(n) == {} and bp.model_options(n) == {}
              for n in G["stage2_compare"]))
    check("presets: halo = the original + the halo of empty_max", bp.sim_options("halo") == bp.HALO and bp.loss_options("halo") == {})
    s3 = bp.GROUPS["stage3_train"]
    check("stage 3: 8 combined variants, each with random brightness and fg3, none scaling inside the model",
          len(s3) == 8 == len(set(s3)) and all(bp.loss_options(n) == {"fg_weight": 3.0} and "gain_log10_range" in bp.sim_options(n)
                                              and bp.model_options(n) == {} for n in s3))
    check("stage 3: pair_max1_aug_* have the paired copies, aug_* not",
          all((bp.data_options(n) == {"max_one_copies": True}) == n.startswith("pair_max1_aug") for n in s3))
    check("stage 3: _halo / _light have the halo / the weaker noise, the others not",
          all(("halo_log10_range" in bp.sim_options(n)) == ("_halo" in n) for n in s3)
          and all((bp.sim_options(n).get("artifact_std_range") == bp.LIGHT_NOISE["artifact_std_range"]) == ("_light" in n) for n in s3))
    check("stage 3: aug_fg3 is aug + fg3 exactly (same generator settings as aug)",
          bp.sim_options("aug_fg3") == bp.sim_options("aug") and bp.hydra_overrides("aug_fg3") == bp.hydra_overrides("aug") + ["+loss.fg_weight=3.0"])
    check("stage 3: compares against the fair original, the two bases and the stage 2 winners",
          set(["original_retrained", "aug", "pair_max1_aug", "fg3"]) <= set(bp.GROUPS["stage3_compare"]) and set(s3) <= set(bp.GROUPS["stage3_compare"]))

    # ---- postprocess
    runs = np.zeros((3, 6, 6), int)
    runs[:2, 2, 2] = 4; runs[2, 2, 2] = 7          # 3 of 3 runs say object, material 4 twice
    runs[0, 4, 4] = 5; runs[1, 4, 4] = 5           # 2 of 3 say object
    runs[0, 0, 0] = 2                              # 1 of 3
    l5, l7, l3 = pp.vote(runs, 0.5), pp.vote(runs, 0.7), pp.vote(runs, 0.3)
    check("postprocess.vote: an object where enough runs say so, the material most runs give",
          l5[2, 2] == 4 and l5[4, 4] == 5 and l5[0, 0] == 0 and l7[4, 4] == 0 and l7[2, 2] == 4 and l3[0, 0] == 2)
    rr = np.random.default_rng(0).integers(0, 6, (4, 10, 10))
    check("postprocess.vote at 0.5 is the notebooks' rule (brightness_eval.combine)", np.array_equal(pp.vote(rr, 0.5), ev.combine(rr)[0]))

    cube = np.zeros((10, 30, 30)); cube[:, 10:20, 10:20] = 1.0
    labels = np.full((30, 30), 3)                                   # the model labels everything as material 3
    masked = pp.apply_outline(labels, cube, "mask@0.01")
    check("postprocess.outline mask: labels outside the outline found in the data are dropped, inside kept",
          masked[15, 15] == 3 and masked[0, 0] == 0 and int((masked > 0).sum()) == 100)
    holey = np.zeros((30, 30), int); holey[10:20, 10:15] = 2; holey[10:20, 15:20] = 0          # right half called background
    filled = pp.apply_outline(holey, cube, "fill@0.01")
    check("postprocess.outline fill: background inside the outline takes the nearest material; outside stays background",
          int((filled > 0).sum()) == 100 and (filled[10:20, 10:20] == 2).all() and filled[0, 0] == 0)
    check("postprocess.outline fill: an outline with nothing labelled in it stays unlabelled", int((pp.apply_outline(np.zeros((30, 30), int), cube, "fill@0.01") > 0).sum()) == 0)
    check("postprocess.outline model: unchanged", np.array_equal(pp.apply_outline(labels, cube, "model"), labels))
    comp = np.zeros((10, 41, 41)); comp[:, 10:26, 10:13] = 1.0; comp[:, 18, 30] = 0.001
    every = np.full((41, 41), 3)
    check("postprocess.outline +faint: the faint object's label is kept (it is dropped without +faint)",
          pp.apply_outline(every, comp, "mask@0.01+faint")[18, 30] == 3 and pp.apply_outline(every, comp, "mask@0.01")[18, 30] == 0)
    one = np.zeros((41, 41), int); one[10:26, 10:13] = 2; one[18, 30] = 4         # the satellite + a 1-pixel companion label
    check("postprocess.finish: tidy keeps a 1-pixel label on a faint object found with +faint (it is not a speck)",
          pp.finish(one, comp, "mask@0.01+faint", "tidy")[18, 30] == 4)
    lone_speck = one.copy(); lone_speck[35, 5] = 4                                  # a 1-pixel label on nothing found
    check("postprocess.finish: tidy still removes a 1-pixel speck that is not on a found object",
          pp.finish(lone_speck, comp, "model", "tidy")[35, 5] == 0)
    try:
        pp.apply_outline(labels, cube, "mask@0.01+bright")
        check("postprocess: rejects an unknown outline extra", False)
    except ValueError:
        check("postprocess: rejects an unknown outline extra", True)
    try:
        pp.apply_outline(labels, cube, "frame@0.01")
        check("postprocess: rejects an unknown outline", False)
    except ValueError:
        check("postprocess: rejects an unknown outline", True)

    lab = np.zeros((20, 20), int); lab[5:12, 5:12] = 2
    lab[8, 8] = 0                                                   # a hole
    lab[17, 17] = 3                                                 # a one-pixel speck
    tidy = pp.cleanup(lab, "tidy")
    check("postprocess.cleanup tidy: removes the speck and fills the hole, leaves the rest", tidy[17, 17] == 0 and tidy[8, 8] == 2 and int((tidy > 0).sum()) == 49)
    mix = np.zeros((20, 20), int); mix[5:12, 5:12] = 2; mix[8, 8] = 4
    smooth = pp.cleanup(mix, "tidy+smooth")
    check("postprocess.cleanup tidy+smooth: a lone odd material in the middle of another takes the majority", smooth[8, 8] == 2 and int((smooth > 0).sum()) == 49)
    check("postprocess.cleanup off: unchanged", np.array_equal(pp.cleanup(lab, "off"), lab))
    try:
        pp.cleanup(lab, "polish")
        check("postprocess: rejects an unknown cleanup", False)
    except ValueError:
        check("postprocess: rejects an unknown cleanup", True)
    check("postprocess.finish: outline, then cleanup", int((pp.finish(labels, cube, "mask@0.01", "tidy") > 0).sum()) == 100)


def check_pairs():
    """data.params.max_one_copies: each training scene twice (as generated + divided by its max); valid/test untouched."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import brightness_presets as bp
    from omegaconf import OmegaConf
    from src.registry import build_dataset, build_model

    def make(variant, n=6):
        sim = {"n_bands": 10, "wavelengths": list(BANDS_10), **bp.sim_options(variant)}
        cfg = OmegaConf.create({"data": {"name": "hsi", "params": {"n_train": n, "n_valid": 4, "n_test": 4, "sim": sim, **bp.data_options(variant)}},
                                "model": {"name": "spectral_spatial", "init_seed": 1, "params": bp.model_options(variant)},
                                "runtime": {"n_channels": 10, "n_classes": 47, "run_idx": 0}})
        return build_dataset(cfg), build_model(cfg)

    ds, model = make("pair_max1_aug")
    tr, va = ds.split("train"), ds.split("valid")
    check("pair_max1: the training split has 2 x n_train items, valid is not doubled", len(tr) == 12 and len(va) == 4)
    res = tr[(0, list(range(12)))]
    ok_same = ok_max = ok_label = ok_ratio = True
    for k in range(6):
        a, b = res[2 * k], res[2 * k + 1]
        ok_same &= a["index"] == b["index"] == k
        ok_label &= torch.equal(a["label"], b["label"])
        ok_max &= abs(float(b["input"].max()) - 1.0) < 1e-6
        ok_ratio &= torch.allclose(b["input"] * a["input"].max(), a["input"], rtol=1e-4, atol=1e-7)
    check("pair_max1: items 2k and 2k+1 are the same scene, same label", ok_same and ok_label)
    check("pair_max1: the second copy's brightest value is exactly 1", ok_max)
    check("pair_max1: the copy is the original times one number (colours and layout unchanged)", ok_ratio)
    check("pair_max1: the 'as generated' copy keeps its own brightness (not all brightest = 1)",
          any(abs(float(res[2 * k]["input"].max()) - 1.0) > 1e-3 for k in range(6)))
    v1, v2 = va[(0, [0, 1])], ds.split("valid")[(0, [0, 1])]
    check("pair_max1: valid scenes are as generated (not rescaled)", float(v1[0]["input"].max()) != 1.0 and torch.equal(v1[0]["input"], v2[0]["input"]))
    check("pair_max1: the model has no input scaling (the rescaling is in the data)", model.input_norm == "none")
    ds0, _ = make("base")
    check("pair_max1: switch off (default) = untouched training split", len(ds0.split("train")) == 6)
    check("pair_max1: hydra override is data.params.max_one_copies=true",
          "+data.params.max_one_copies=true" in bp.hydra_overrides("pair_max1") and not any("input_norm" in o for o in bp.hydra_overrides("pair_max1")))

    # the second copy divided by the mean / the 99th percentile of the object pixels
    from src.datasets.hsi import copy_scale
    for stat in ("mean", "p99"):
        ds2, model2 = make(f"pair_{stat}")
        res2 = ds2.split("train")[(0, list(range(12)))]
        ok_one = ok_ratio = ok_label = True
        for k in range(6):
            a, b = res2[2 * k], res2[2 * k + 1]
            obj = a["label"] > 0
            vals = (b["input"][:, obj] if bool(obj.any()) else b["input"].reshape(10, -1)).flatten()
            v = vals.mean() if stat == "mean" else torch.quantile(vals, 0.99)
            ok_one &= abs(float(v) - 1.0) < 1e-4
            ok_label &= torch.equal(a["label"], b["label"])
            ok_ratio &= torch.allclose(b["input"] * copy_scale(a["input"], a["label"], stat), a["input"], rtol=1e-4, atol=1e-6)
        check(f"pair_{stat}: in the second copy the {stat} of the object pixels is 1; same label; the copy is the original times one number",
              ok_one and ok_label and ok_ratio)
        check(f"pair_{stat}: no scaling in the model, hydra override data.params.pair_copies={stat}",
              model2.input_norm == "none" and f"+data.params.pair_copies={stat}" in bp.hydra_overrides(f"pair_{stat}"))
    try:
        bad_sim = {"n_bands": 10, "wavelengths": list(BANDS_10)}
        bad = OmegaConf.create({"data": {"name": "hsi", "params": {"n_train": 2, "n_valid": 2, "n_test": 2, "sim": bad_sim, "pair_copies": "median"}},
                                "model": {"name": "spectral_spatial", "init_seed": 1, "params": {}},
                                "runtime": {"n_channels": 10, "n_classes": 47, "run_idx": 0}})
        build_dataset(bad).split("train")
        check("pair_copies: rejects an unknown divisor", False)
    except ValueError:
        check("pair_copies: rejects an unknown divisor", True)

    # the 24 pair variants: every combination of divisor x random brightness x halo / weaker noise
    pv = bp.PAIR_VARIANTS
    check("presets: 24 pair variants (3 divisors x brightness or not x nothing / halo / weaker noise / both), no scaling in the model",
          len(pv) == 24 == len(set(pv)) and all(bp.model_options(n) == {} for n in pv)
          and all(bool(bp.data_options(n)) for n in pv))
    check("presets: _aug pair variants have the random brightness, the others not",
          all(("gain_log10_range" in bp.sim_options(n)) == ("_aug" in n) for n in pv))
    check("presets: _halo / _light pair variants have the halo / the weaker noise, the others not",
          all(("halo_log10_range" in bp.sim_options(n)) == ("_halo" in n) for n in pv)
          and all((bp.sim_options(n).get("artifact_std_range") == bp.LIGHT_NOISE["artifact_std_range"]) == ("_light" in n) for n in pv))
    check("presets: pair_max1 and pair_max1_aug are unchanged by the new grid",
          bp.data_options("pair_max1") == {"max_one_copies": True} and bp.sim_options("pair_max1") == {}
          and set(bp.sim_options("pair_max1_aug")) == {"gain_log10_range", "gain_stage"})
    check("presets: the pipeline trains the 24 pair variants and fg3, fg10, fg3_dice, lightnoise",
          bp.PIPELINE == pv + ["fg3", "fg10", "fg3_dice", "lightnoise"])


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

    check_fits_reading()
    check_presets()
    check_pairs()
    check_auto_scale()
    check_pipeline_parts()

    print("\nALL PASS" if not FAILED else f"\n{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
