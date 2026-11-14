"""The named variants of the brightness experiments: ONE place that says what each one is.

slurm/train_brightness.sh asks this file for the Hydra overrides, scripts/brightness_quick_train.py
builds its config from it, and scripts/brightness_compare.py reads the labels from it, so the
cluster runs, the CPU runs and the comparison cannot drift apart. Standard library only.

    python scripts/brightness_presets.py list
    python scripts/brightness_presets.py hydra full_max [--gain-range -1.5,0.5] [--gain-stage before_noise]

An experiment's folder is logs/estimate_10band_<preset>  (`base` is plain logs/estimate_10band).
"""
import argparse
import os
import sys

GAIN = [-1.5, 0.5]            # log10 of the overall brightness factor: 0.03x to 3x

# smooth random curve over the bands, standing in for sunlight x atmosphere x optics x detector
CURVE = {"response_log10_std": 0.2, "response_knots": 4}

# scenes with no object, a faint skirt around objects (labelled background), restoration-like noise:
# three separate parts, so that each can be tried on its own (add_empty, add_halo, add_noise)
EMPTY_SCENES = {"empty_prob": 0.15, "empty_noise_log10_range": [-3.0, 0.0]}      # 15% of scenes hold no object
HALO = {"halo_log10_range": [-4.0, -1.5], "halo_sigma_px": [2.0, 6.0]}           # faint skirt round the object = background
NOISE = {"artifact_std_range": [0.002, 0.03]}                                    # smooth, skewed, band-correlated noise
EMPTY = {**EMPTY_SCENES, **HALO, **NOISE}                                        # all three: what empty_max and full_* use
LIGHT_NOISE = {"artifact_std_range": [0.001, 0.008]}                              # the same kind of noise, 4x weaker at the top


def preset(description, gain=False, norm="none", pairs=None, loss=None, bank=False, **sim):
    """pairs: train on every scene twice, as generated and divided by one number: "max" = its brightest value
    (data.params.max_one_copies), "mean" / "p99" = the mean / 99th percentile of its object pixels (data.params.pair_copies).
    loss: {"fg_weight": .., "dice": ..} for src/losses.py. bank: add the noise bank named by the environment variable
    NOISE_BANK (a .npy made with scripts/make_noise_bank.py)."""
    return {"description": description, "gain": gain, "sim": sim, "model": {"input_norm": norm} if norm != "none" else {},
            "data": {} if not pairs else ({"max_one_copies": True} if pairs == "max" else {"pair_copies": pairs}),
            "loss": dict(loss or {}), "bank": bank}


PRESETS = {
    # ---- the four of the first round
    "base":      preset("the original model and data"),
    # the original again, trained HERE with exactly the setup and seeds of the new variants (1 GPU, slurm/train_brightness.sh).
    # The `base` folder (logs/estimate_10band) was trained elsewhere on 2 GPUs, so it is not a fair reference for them.
    # fg_weight=1 is the plain cross-entropy (src/losses.py); it is there only so that the training script has an override to pass.
    "original_retrained": preset("the original model and data, retrained with the same setup and seeds as the new variants",
                                 loss={"fg_weight": 1.0}),
    "aug":       preset("random overall brightness in training; raw values", gain=True),
    "norm":      preset("divide by the brightest value inside the model; ordinary training", norm="max"),
    "aug_norm":  preset("random brightness + divide by the max (the 'both' of round 1)", gain=True, norm="max"),
    # ---- other ways to normalize (all trained with random brightness)
    "gain_p99":      preset("random brightness + divide by the 99th percentile", gain=True, norm="p99"),
    "gain_flux":     preset("random brightness + divide by the mean of all values (total flux)", gain=True, norm="flux"),
    "gain_bandflux": preset("random brightness + divide each band by its own mean", gain=True, norm="band_flux"),
    "gain_pixel":    preset("random brightness + divide each pixel by its own brightest band (shape only)", gain=True, norm="pixel"),
    # ---- robustness add-ons
    "curve_max": preset("aug_norm + a random smooth curve over the bands", gain=True, norm="max", **CURVE),
    "empty_max": preset("aug_norm + empty scenes, halo and restoration-like noise", gain=True, norm="max", **EMPTY),
    # ---- the three parts of empty_max, one at a time (which of them makes the model drop faint real structure?)
    "add_empty": preset("aug_norm + ONLY empty scenes (15% of scenes have no object)", gain=True, norm="max", **EMPTY_SCENES),
    "add_halo":  preset("aug_norm + ONLY the halo (a faint skirt around the object, labelled background)", gain=True, norm="max", **HALO),
    "add_noise": preset("aug_norm + ONLY restoration-like noise (smooth, skewed, band-correlated)", gain=True, norm="max", **NOISE),
    # ---- the shape of the object: a heavier loss on object pixels, less or real noise (all on the ORIGINAL data and model:
    #      no brightness augmentation, no scaling inside; the brightness is handled in front of the model, scripts/auto_scale.py)
    "fg3":        preset("original training, every object pixel counts 3x in the loss", loss={"fg_weight": 3.0}),
    "fg10":       preset("original training, every object pixel counts 10x in the loss", loss={"fg_weight": 10.0}),
    "fg3_dice":   preset("original training, object pixels 3x + a Dice term for object vs background", loss={"fg_weight": 3.0, "dice": 0.5}),
    "lightnoise": preset("original training + weaker restoration-like noise (std 0.001-0.008 of the object)", **LIGHT_NOISE),
    "halo":       preset("original training + the halo (a faint skirt round the object, labelled background)", **HALO),
    "bank":       preset("original training + real noise patches from the noise bank in $NOISE_BANK", bank=True),
    "full_max":  preset("aug_norm + curve + empty scenes, halo, artifacts", gain=True, norm="max", **CURVE, **EMPTY),
    "full_p99":  preset("the same, normalized by the 99th percentile", gain=True, norm="p99", **CURVE, **EMPTY),
    "full_raw":  preset("the same, with NO normalization (augmentation alone)", gain=True, **CURVE, **EMPTY),
}

# ---- the normalization done to the TRAINING DATA, not inside the model: each scene twice, as generated and divided by
#      one number. Every combination of: the number (max / mean / p99 of the object), random brightness on the
#      "as generated" copy or not, and the halo and/or the weaker restoration-like noise. 3 x 2 x 4 = 24 variants:
#      pair_<max1|mean|p99>[_aug][_halo][_light]
PAIR_STATS = {"max": ("max1", "its brightest value (brightest = 1)"),
              "mean": ("mean", "the mean of its object pixels (object mean = 1)"),
              "p99": ("p99", "the 99th percentile of its object pixels")}
PAIR_EXTRAS = {"": ({}, ""), "_halo": (HALO, " + the halo"), "_light": (LIGHT_NOISE, " + weaker restoration-like noise"),
               "_halo_light": ({**HALO, **LIGHT_NOISE}, " + the halo + weaker restoration-like noise")}
PAIR_VARIANTS = []
for _stat, (_tag, _what) in PAIR_STATS.items():
    for _aug in (False, True):
        for _suffix, (_sim, _extra) in PAIR_EXTRAS.items():
            _name = f"pair_{_tag}{'_aug' if _aug else ''}{_suffix}"
            PRESETS[_name] = preset(f"every training scene twice: as generated{' (random brightness)' if _aug else ''} and divided by "
                                    f"{_what}{_extra}; no scaling in the model", gain=_aug, pairs=_stat, **_sim)
            PAIR_VARIANTS.append(_name)

# the variants slurm/run_pipeline.sh trains by default (the pair variants and the shape variants; `bank` only with $NOISE_BANK)
PIPELINE = PAIR_VARIANTS + ["fg3", "fg10", "fg3_dice", "lightnoise"]

# ---- stage 3: the best of stage 1 (brightness) and stage 2 (shape) together. Two brightness bases (`aug`: random
#      brightness; `pair_max1_aug`: random brightness + every scene twice, the copy divided by its max) x the shape
#      winner fg3 (object pixels 3x in the loss), alone or with the halo and/or the weaker restoration-like noise.
FG3 = {"fg_weight": 3.0}
STAGE3 = []
for _base, _bwhat, _pairs in (("aug", "random brightness", None), ("pair_max1_aug", "random brightness + every scene twice (copy brightest = 1)", "max")):
    for _suffix, (_sim, _extra) in PAIR_EXTRAS.items():
        _name = f"{_base}_fg3{_suffix}"
        PRESETS[_name] = preset(f"{_bwhat} + object pixels 3x in the loss{_extra}", gain=True, pairs=_pairs, loss=FG3, **_sim)
        STAGE3.append(_name)

# the two separate stages (slurm/run_brightness_stage.sh, slurm/run_shape_stage.sh):
#   *_train    the variants each stage trains
#   *_compare  what each stage scores: its new variants and the earlier models it is compared with (already trained)
# Stage 1 changes ONLY the brightness handling (no halo, no extra noise, the plain loss). Stage 2 keeps the brightness
# handling fixed (the original training; automatic max scaling at test time) and changes only the shape choices.
GROUPS = {
    "stage1_train": ["pair_max1", "pair_mean", "pair_p99", "pair_max1_aug", "pair_mean_aug", "pair_p99_aug"],
    "stage1_compare": ["base", "original_retrained", "aug", "norm", "aug_norm", "gain_p99", "gain_flux", "gain_bandflux", "gain_pixel",
                       "pair_max1", "pair_mean", "pair_p99", "pair_max1_aug", "pair_mean_aug", "pair_p99_aug"],
    "stage2_train": ["fg3", "fg10", "fg3_dice", "halo", "lightnoise"],
    "stage2_compare": ["base", "original_retrained", "fg3", "fg10", "fg3_dice", "halo", "lightnoise", "bank"],
    "reference_train": ["original_retrained"],      # train once; both stages compare against it
    "stage3_train": STAGE3,
    "stage3_compare": ["original_retrained", "aug", "pair_max1_aug", "fg3", "halo", "lightnoise"] + STAGE3,
    "pipeline": PIPELINE,
}

# the order a comparison is drawn in
ORDER = list(PRESETS)


def experiment_name(name):
    return "estimate_10band" if name == "base" else f"estimate_10band_{name}"


def label(name):
    return {"base": "original"}.get(name, name)


def sim_options(name, gain_range=None, gain_stage="before_noise"):
    """The SimConfig keys of a preset (lists stay lists). A `bank` preset adds $NOISE_BANK when that is set."""
    p = PRESETS[name]
    sim = dict(p["sim"])
    if p["bank"] and os.environ.get("NOISE_BANK"):
        sim["noise_bank"] = os.environ["NOISE_BANK"]
    if p["gain"]:
        sim["gain_log10_range"] = list(gain_range or GAIN)
        sim["gain_stage"] = gain_stage
    return sim


def model_options(name):
    return dict(PRESETS[name]["model"])


def loss_options(name):
    """Keys of the `loss` section (src/losses.py); empty = the plain cross-entropy."""
    return dict(PRESETS[name]["loss"])


def data_options(name):
    """Keys of data.params (not of sim)."""
    return dict(PRESETS[name]["data"])


def _hydra_value(v):
    if isinstance(v, bool):
        return str(v).lower()
    return "[" + ",".join(repr(float(x)) for x in v) + "]" if isinstance(v, (list, tuple)) else str(v)


def hydra_overrides(name, gain_range=None, gain_stage="before_noise"):
    if PRESETS[name]["bank"] and not os.environ.get("NOISE_BANK"):
        sys.exit(f"variant '{name}' needs the environment variable NOISE_BANK (a .npy noise bank, scripts/make_noise_bank.py)")
    args = [f"+data.params.sim.{k}={_hydra_value(v)}" for k, v in sim_options(name, gain_range, gain_stage).items()]
    args += [f"+data.params.{k}={_hydra_value(v)}" for k, v in data_options(name).items()]
    args += [f"+loss.{k}={_hydra_value(v)}" for k, v in loss_options(name).items()]
    args += [f"+model.params.{k}={_hydra_value(v)}" for k, v in model_options(name).items()]
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("pipeline", help="print the variants slurm/run_pipeline.sh trains by default")
    g = sub.add_parser("group", help="print a group of variants: " + ", ".join(GROUPS))
    g.add_argument("name", choices=list(GROUPS))
    g.add_argument("--folders", action="store_true", help="print the experiment folders instead of the variant names")
    h = sub.add_parser("hydra")
    h.add_argument("name")
    h.add_argument("--gain-range", default=None, help="log10 'lo,hi' (default -1.5,0.5)")
    h.add_argument("--gain-stage", default="before_noise")
    a = ap.parse_args()
    if a.cmd == "pipeline":
        print(" ".join(PIPELINE))
        return
    if a.cmd == "group":
        print(" ".join(experiment_name(n) if a.folders else n for n in GROUPS[a.name]))
        return
    if a.cmd == "list":
        for n, p in PRESETS.items():
            print(f"{n:14s} {experiment_name(n):34s} {p['description']}")
        return
    if a.name not in PRESETS:
        sys.exit(f"unknown variant '{a.name}'; one of: {', '.join(PRESETS)}")
    rng = [float(v) for v in a.gain_range.split(",")] if a.gain_range else None
    print(" ".join(hydra_overrides(a.name, rng, a.gain_stage)))


if __name__ == "__main__":
    main()
