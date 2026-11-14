"""Which way of handling brightness works best? Models x automatic scaling, on estimate.fits and on synthetic scenes.

Models (folders under --logdir; the four below and fg3, fg10, fg3_dice, lightnoise, bank, see BRIGHTNESS.md; one without
a folder is skipped):

    estimate_10band               the original model, trained on scenes as generated
    estimate_10band_pair_max1     every training scene twice: as generated + divided by its max (brightest = 1)
    estimate_10band_pair_max1_aug the same, with a random overall brightness on the "as generated" copy
    estimate_10band_aug_norm      random brightness + the model divides by the max itself (reference)

Scaling in front of the model (scripts/auto_scale.py, nothing learned): none | single | per_object, and single / per_object
by the mean or the 99th percentile of the object (single_mean, single_p99, per_object_mean, per_object_p99).
Every model is scored with every mode, so "which training" and "which scaling" can be read separately.

Tests (all from the same models):
  1. estimate.fits against object.fits, the 41 px window round satellite and companion (as notebook 06), per reading
     (solar = notebook 06 / radiometry = the collaborators' recipe). Next to the modes, the notebook's by-hand rows
     as the reference: "satellite max = 1 by hand", "companion on its own scale by hand". A window with nothing in it.
  2. Synthetic satellite scenes at many overall brightnesses (the whole image rescaled, or a dimmer object against the
     same noise): do the models, with each mode, give the same answer whatever the brightness?
  3. Synthetic TWO-OBJECT scenes: a bright and a dim satellite in one image (the dim one 1x, 0.3x, 0.1x, 0.03x as bright).
     This is the case that one scale number cannot fit: accuracy is reported for the bright and the dim object separately.

Run from the project root, after the models are trained (nothing here trains):

    python scripts/compare_scaling.py
    python scripts/compare_scaling.py --experiments estimate_10band estimate_10band_pair_max1 --readings radiometry

Writes to --out (default outputs/scaling_compare): scaling_fits.csv / .md, fits_maps_<reading>.png,
scaling_single_object.csv, scaling_two_objects.csv, and scaling_summary.md with the tables.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import auto_scale  # noqa: E402
import brightness_eval as ev  # noqa: E402
from brightness_presets import PRESETS, experiment_name  # noqa: E402
from src.data_generation import generate_sample  # noqa: E402

MODES = auto_scale.MODES
DEFAULT_EXPERIMENTS = [experiment_name(n) for n in PRESETS]      # every variant there is; the ones not trained are skipped
BY_HAND = ("satellite max = 1 by hand", "companion on its own scale by hand")
FITS_METRICS = ["outline_iou", "ari_materials", "false_px", "labelled_px", "satellite_iou", "companion_px_found"]
SINGLE_GAINS = {"before_noise": [-2.0, -1.5, -1.0, -0.5, 0.0], "after_noise": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]}
DIM_LEVELS = [0.0, -0.5, -1.0, -1.5]           # log10 of the dim object's brightness against the bright one


def run_mode(exp, cube, mode):
    """cube [bands, H, W] -> labels [runs, H, W], the model's answer when the image is scaled as `mode` says."""
    return auto_scale.predict_scaled(lambda x: exp.predict(x), cube, mode, len(exp.models))


# ----------------------------------------------------------------------------- 1. estimate.fits
def window_row(scene, labels, objectness, agreement, size, centre):
    """The scores of ev.fits_results for one window (same definitions, kept in step with it by hand)."""
    found = labels > 0
    tf, tr = scene.win(scene.true_fg, size, centre), scene.win(scene.regions, size, centre)
    ts, tc = scene.win(scene.true_sat, size, centre), scene.win(scene.true_comp, size, centre)
    return {"window_px": size, "labelled_px": int(found.sum()), "false_px": int((found & ~tf).sum()),
            "outline_iou": (found & tf).sum() / max((found | tf).sum(), 1) if tf.any() else np.nan,
            "ari_materials": ev.ari(tr[tf], labels[tf]) if tf.sum() > 1 else np.nan,
            "satellite_iou": (found & ts).sum() / max((found | ts).sum(), 1) if ts.any() else np.nan,
            "companion_px_found": int((found & tc).sum()), "companion_px_true": int(tc.sum())}


def fits_part(exps, readings, out):
    lam = exps[0].data.cfg.wavelengths_nm
    frames = []
    for reading in readings:
        scene = ev.FitsScene(lam, reading)
        size, centre = scene.SIZE, scene.centre
        windows = {"both objects": (scene.win(scene.reading, size, centre).copy(), centre),
                   "blank window": (scene.win(scene.reading, size, scene.blank_centre).copy(), scene.blank_centre)}
        rows, maps = [], {}
        for e in exps:
            for mode in MODES:
                for wname, (cube, c) in windows.items():
                    labels, objectness, agreement = ev.combine(run_mode(e, cube, mode))
                    row = window_row(scene, labels, objectness, agreement, size, c)
                    if wname == "blank window":           # nothing in it: every labelled pixel is false
                        row.update(false_px=row["labelled_px"], outline_iou=np.nan, ari_materials=np.nan,
                                   satellite_iou=np.nan, companion_px_found=0)
                    rows.append({"reading": reading, "experiment": e.name, "mode": mode, "window": wname, **row})
                    if wname == "both objects":
                        maps[(e.name, mode)] = labels
        by_hand, hand_maps = ev.fits_results(exps, scene)          # the notebook's by-hand rows, as the reference
        by_hand = by_hand[by_hand.setting.isin(BY_HAND)]
        for _, r in by_hand.iterrows():
            rows.append({**r.to_dict(), "reading": reading, "mode": r.setting, "window": "both objects"})
            maps[(r.experiment, r.setting)] = hand_maps[(r.experiment, r.setting)]["labels"]
        frame = pd.DataFrame(rows)
        frames.append(frame)
        plot_maps(scene, exps, maps, out / f"fits_maps_{reading}.png", reading)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "scaling_fits.csv", index=False)
    return df


def plot_maps(scene, exps, maps, path, reading):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data_generation import N_CLASSES

    cols = [("truth", None)] + [(m, m) for m in MODES] + [(s, s) for s in BY_HAND]
    fig, axes = plt.subplots(len(exps), len(cols), figsize=(2.3 * len(cols), 2.4 * len(exps)), squeeze=False)
    truth = scene.win(scene.regions, scene.SIZE, scene.centre)
    cmap = plt.get_cmap("nipy_spectral").copy()
    cmap.set_bad("white")
    for i, e in enumerate(exps):
        for j, (title, key) in enumerate(cols):
            ax = axes[i][j]
            lab = truth if key is None else maps.get((e.name, key))
            ax.axis("off")
            if lab is None:
                continue
            ax.imshow(np.ma.masked_equal(lab, 0), cmap=cmap, vmin=1, vmax=N_CLASSES, interpolation="nearest")
            if i == 0:
                ax.set_title(title.replace(" by hand", "\nby hand"), fontsize=7)
            if j == 0:
                ax.text(-0.05, 0.5, e.label, transform=ax.transAxes, ha="right", va="center", fontsize=7, rotation=90)
    fig.suptitle(f"estimate.fits, reading '{reading}': material labels (white = background)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- 2. one object, many brightnesses
def pooled(pred, mask):
    """material_acc, foreground_iou, false_fg_share of predictions [N, H, W] against masks [N, H, W]."""
    fg, found = mask > 0, pred > 0
    return {"material_acc": float((pred == mask)[fg].mean()),
            "foreground_iou": float((found & fg).sum() / max((found | fg).sum(), 1)),
            "false_fg_share": float(found[~fg].mean())}


def single_part(exps, n_scenes, seed, out):
    ref = ev.check_same_bands(exps)
    rows = []
    for stage, gains in SINGLE_GAINS.items():
        cfg = ev.clean_cfg(ref.cfg).with_(gain_stage=stage)
        for lg in gains:
            samples = [generate_sample(cfg, ev.scene_seed(seed, i), ref.library, scene="satellite", gain=10.0 ** lg)
                       for i in range(n_scenes)]
            masks = np.stack([s.mask for s in samples])
            for e in exps:
                for mode in MODES:
                    preds = np.stack([run_mode(e, s.cube, mode) for s in samples], 1)        # [runs, N, H, W]
                    for r in range(preds.shape[0]):
                        rows.append({"experiment": e.name, "mode": mode, "run": r, "stage": stage, "log10_gain": lg,
                                     **pooled(preds[r], masks)})
            print(f"  single object  {stage:12s} log10 gain {lg:+.1f} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "scaling_single_object.csv", index=False)
    return df


# ----------------------------------------------------------------------------- 3. a bright and a dim object in one image
def two_object_part(exps, n_scenes, seed, out):
    ref = ev.check_same_bands(exps)
    cfg = ev.clean_cfg(ref.cfg).with_(gain_stage="before_noise")        # dimmer object, the same noise
    rows = []
    for lg in DIM_LEVELS:
        cubes, masks, dim_cols = [], [], []
        for i in range(n_scenes):
            bright = generate_sample(cfg, ev.scene_seed(seed, 2, i, 0), ref.library, scene="satellite")
            dim = generate_sample(cfg, ev.scene_seed(seed, 2, i, 1), ref.library, scene="satellite", gain=10.0 ** lg)
            first, second = (bright, dim) if i % 2 == 0 else (dim, bright)         # the dim one on either side
            w = first.cube.shape[-1]
            cubes.append(np.concatenate([first.cube, second.cube], axis=-1))
            masks.append(np.concatenate([first.mask, second.mask], axis=-1))
            cols = np.zeros(masks[-1].shape, bool)
            if i % 2 == 0:
                cols[:, w:] = True
            else:
                cols[:, :w] = True
            dim_cols.append(cols)
        masks, dim_cols = np.stack(masks), np.stack(dim_cols)
        for e in exps:
            for mode in MODES:
                preds = np.stack([run_mode(e, c, mode) for c in cubes], 1)             # [runs, N, H, 2W]
                for r in range(preds.shape[0]):
                    fg = masks > 0
                    acc = lambda sel: float((preds[r] == masks)[fg & sel].mean()) if (fg & sel).any() else np.nan   # noqa: E731
                    rows.append({"experiment": e.name, "mode": mode, "run": r, "log10_dim": lg,
                                 "acc_bright": acc(~dim_cols), "acc_dim": acc(dim_cols), **pooled(preds[r], masks)})
        print(f"  two objects  dim one at log10 {lg:+.1f} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "scaling_two_objects.csv", index=False)
    return df


# ----------------------------------------------------------------------------- the summary
def md_table(df):
    """A pandas table as markdown (no extra package needed)."""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    fmt = lambda v: "" if isinstance(v, (float, np.floating)) and np.isnan(v) else (  # noqa: E731
        f"{v:g}" if isinstance(v, (float, np.floating)) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def table(df, index, columns, values, digits=3):
    return df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean").round(digits)


def write_summary(out, exps, fits, single, two):
    short = {e.name: e.label for e in exps}
    md = ["# Brightness: training x scaling", "",
          "Models: " + ", ".join(f"`{e.label}`" for e in exps) + ". Modes (scripts/auto_scale.py): `none` = the image as it is, "
          "`single` = one scale for the image, `per_object` = every object its own scale. The by-hand rows are the notebook's "
          "reference (a person picked the objects and the scales).", ""]
    if fits is not None:
        for reading in fits.reading.unique():
            part = fits[(fits.reading == reading) & (fits.window == "both objects")].assign(model=lambda d: d.experiment.map(short))
            md += [f"## estimate.fits, reading `{reading}`", "", "Higher is better for iou / ari / companion_px_found; lower for false_px. "
                   "The true companion is 1 pixel, 18 px right of the satellite.", ""]
            md += [md_table(part.set_index(["model", "mode"])[FITS_METRICS].round(2)), ""]
            blank = fits[(fits.reading == reading) & (fits.window == "blank window")].assign(model=lambda d: d.experiment.map(short))
            md += ["Window with nothing in it (every labelled pixel is a false one):", "",
                   md_table(blank.pivot_table(index="model", columns="mode", values="labelled_px").round(0)), ""]
    if single is not None:
        md += ["## One object, overall brightness changed (material accuracy; flat rows = brightness does not matter)", ""]
        for stage in single.stage.unique():
            t = table(single[single.stage == stage].assign(model=lambda d: d.experiment.map(short)),
                      ["model", "mode"], "log10_gain", "material_acc")
            md += [f"`{stage}`, columns = log10 of the gain:", "", md_table(t), ""]
    if two is not None:
        md += ["## A bright and a dim object in one image (columns = log10 of the dim object's brightness)", ""]
        for name in ("acc_bright", "acc_dim", "false_fg_share"):
            t = table(two.assign(model=two.experiment.map(short)), ["model", "mode"], "log10_dim", name)
            md += [f"`{name}`:", "", md_table(t), ""]
    (out / "scaling_summary.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=None, help="folders under --logdir (default: the four above)")
    ap.add_argument("--logdir", default=str(ROOT / "logs"))
    ap.add_argument("--readings", nargs="+", default=["solar", "radiometry"], choices=("solar", "radiometry", "radiometry_sun"))
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--n-scenes", type=int, default=100, help="synthetic scenes per level")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--modes", nargs="+", default=list(auto_scale.MODES), choices=auto_scale.MODES,
                    help="test-time scalings to try (default: all)")
    ap.add_argument("--skip-fits", action="store_true")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs/scaling_compare"))
    args = ap.parse_args()
    global MODES
    MODES = tuple(args.modes)                         # every part below reads the modes from here

    logdir = Path(args.logdir)
    names = args.experiments or DEFAULT_EXPERIMENTS
    have = [n for n in names if (logdir / n / "config.yaml").is_file()]
    for n in set(names) - set(have):
        print(f"skipping {n}: no folder with a config.yaml under {logdir}", flush=True)
    if not have:
        sys.exit(f"none of the experiments exist under {logdir}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exps = [ev.Experiment(logdir, n, args.max_runs) for n in have]
    for e in exps:
        print(f"{e.name}: {len(e.models)} runs, input_norm={e.norm}", flush=True)

    fits = single = two = None
    if not args.skip_fits:
        print("estimate.fits", flush=True)
        fits = fits_part(exps, args.readings, out)
    if not args.skip_synthetic:
        print("synthetic, one object", flush=True)
        single = single_part(exps, args.n_scenes, args.seed, out)
        print("synthetic, a bright and a dim object", flush=True)
        two = two_object_part(exps, args.n_scenes, args.seed, out)
    write_summary(out, exps, fits, single, two)
    print("wrote", out / "scaling_summary.md")


if __name__ == "__main__":
    main()
