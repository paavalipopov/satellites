"""The named variants of the brightness experiments: ONE place that says what each one is.

slurm/train_brightness.sh asks this file for the Hydra overrides, scripts/brightness_quick_train.py
builds its config from it, and scripts/brightness_compare.py reads the labels from it, so the
cluster runs, the CPU runs and the comparison cannot drift apart. Standard library only.

    python scripts/brightness_presets.py list
    python scripts/brightness_presets.py hydra full_max [--gain-range -1.5,0.5] [--gain-stage before_noise]

An experiment's folder is logs/estimate_10band_<preset>  (`base` is plain logs/estimate_10band).
"""
import argparse
import sys

GAIN = [-1.5, 0.5]            # log10 of the overall brightness factor: 0.03x to 3x

# smooth random curve over the bands, standing in for sunlight x atmosphere x optics x detector
CURVE = {"response_log10_std": 0.2, "response_knots": 4}

# scenes with no object, a faint skirt around objects (labelled background), restoration-like noise
EMPTY = {"empty_prob": 0.15, "empty_noise_log10_range": [-3.0, 0.0],
         "halo_log10_range": [-4.0, -1.5], "halo_sigma_px": [2.0, 6.0],
         "artifact_std_range": [0.002, 0.03]}


def preset(description, gain=False, norm="none", **sim):
    return {"description": description, "gain": gain, "sim": sim, "model": {"input_norm": norm} if norm != "none" else {}}


PRESETS = {
    # ---- the four of the first round
    "base":      preset("the original model and data"),
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
    "full_max":  preset("aug_norm + curve + empty scenes, halo, artifacts", gain=True, norm="max", **CURVE, **EMPTY),
    "full_p99":  preset("the same, normalized by the 99th percentile", gain=True, norm="p99", **CURVE, **EMPTY),
    "full_raw":  preset("the same, with NO normalization (augmentation alone)", gain=True, **CURVE, **EMPTY),
}

# the order a comparison is drawn in
ORDER = list(PRESETS)


def experiment_name(name):
    return "estimate_10band" if name == "base" else f"estimate_10band_{name}"


def label(name):
    return {"base": "original"}.get(name, name)


def sim_options(name, gain_range=None, gain_stage="before_noise"):
    """The SimConfig keys of a preset (lists stay lists)."""
    p = PRESETS[name]
    sim = dict(p["sim"])
    if p["gain"]:
        sim["gain_log10_range"] = list(gain_range or GAIN)
        sim["gain_stage"] = gain_stage
    return sim


def model_options(name):
    return dict(PRESETS[name]["model"])


def _hydra_value(v):
    return "[" + ",".join(repr(float(x)) for x in v) + "]" if isinstance(v, (list, tuple)) else str(v)


def hydra_overrides(name, gain_range=None, gain_stage="before_noise"):
    args = [f"+data.params.sim.{k}={_hydra_value(v)}" for k, v in sim_options(name, gain_range, gain_stage).items()]
    args += [f"+model.params.{k}={_hydra_value(v)}" for k, v in model_options(name).items()]
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    h = sub.add_parser("hydra")
    h.add_argument("name")
    h.add_argument("--gain-range", default=None, help="log10 'lo,hi' (default -1.5,0.5)")
    h.add_argument("--gain-stage", default="before_noise")
    a = ap.parse_args()
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
