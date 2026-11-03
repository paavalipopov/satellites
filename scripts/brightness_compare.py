"""Compare the new experiments with the previous (original) model: accuracy tables AND the maps.

For every experiment it draws what notebook 06 draws for `estimate_10band` -- the true materials of
object.fits, the models' labels, the share of models that call each pixel an object, and the share
that agree on its material -- in three versions: the satellite's max set to 1 by hand (notebook 06's
best case), NO scale chosen (what the model gets on its own), and the companion on its own scale. Then
it puts all experiments side by side, and writes one table of accuracies against the original.

    python scripts/brightness_compare.py                      # the original + every preset folder found in logs/
    python scripts/brightness_compare.py --baseline estimate_10band \\
        --experiments estimate_10band_aug_norm estimate_10band_full_max
    python scripts/brightness_compare.py --n-scenes 50 --max-runs 3      # a quicker look

On the cluster: sbatch slurm/compare_brightness.sh   (or: bash slurm/compare_brightness.sh)

Outputs in --out (default outputs/brightness_compare):
    comparison_summary.csv / .md   one row per experiment, the accuracies that matter, and the change from the original
    maps_side_by_side.png          every experiment, by-hand and no-scale labels, next to the truth
    maps_<experiment>.png          the notebook-06 figure for each experiment
    synthetic_vs_gain.png, robustness.png, and the csv behind every number
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import brightness_eval as ev  # noqa: E402
from brightness_presets import ORDER, PRESETS, experiment_name  # noqa: E402
from src.data_generation import N_CLASSES  # noqa: E402

# the colours of notebook 06
INK, INK2, GRID, TRUE = "#0b0b0b", "#52514e", "#e4e4e2", "#00b050"

# (setting in brightness_eval.FitsScene.settings, the row's name in the figures)
MAP_ROWS = [("satellite max = 1 by hand", "by hand: satellite max = 1"),
            ("no scale chosen", "no scale chosen"),
            ("companion on its own scale by hand", "companion on its own scale (by hand)")]


# ----------------------------------------------------------------------------- the maps
def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    from matplotlib.colors import ListedColormap, hsv_to_rgb
    hues = [((i * 0.618) % 1, 0.85, 0.95 if i % 2 else 0.7) for i in range(N_CLASSES - 1)]
    materials = ListedColormap(["white", *(hsv_to_rgb(h) for h in hues)])                      # our 46 materials
    regions = ListedColormap(["white", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"])  # true A-E
    return materials, regions


def show(ax, image, title="", **kw):
    im = ax.imshow(image, interpolation="nearest", **kw)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(True)
    return im


def outline(ax, mask):
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=TRUE, linewidths=1)


def plot_maps(exp_label, maps, path):
    """Notebook 06's figure for one experiment: the truth, then three rows of labels / share object / share material."""
    import matplotlib.pyplot as plt
    materials, regions = style()
    first = maps[MAP_ROWS[0][0]]
    names = [[f"{k} {r}" for k in ("labels", "object", "material")] for r in range(len(MAP_ROWS))]
    mosaic = [["truth", *row] for row in names]
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(12, 2.9 * len(MAP_ROWS)), width_ratios=[1.5, 1, 1, 1], layout="constrained")
    show(ax["truth"], first["regions"], "true materials (object.fits)", cmap=regions, vmin=0, vmax=5)
    im = None
    for r, (key, title) in enumerate(MAP_ROWS):
        m = maps[key]
        top = r == 0
        show(ax[f"labels {r}"], m["labels"], "models' labels" if top else "", cmap=materials, vmin=0, vmax=N_CLASSES - 1)
        im = show(ax[f"object {r}"], m["objectness"], "share of models: an object" if top else "", cmap="Blues", vmin=0, vmax=1)
        show(ax[f"material {r}"], np.where(m["labels"] > 0, m["agreement"], 0), "share of models: its material" if top else "",
             cmap="Blues", vmin=0, vmax=1)
        ax[f"labels {r}"].set_ylabel(title, fontsize=9)
        for kind in ("labels", "object", "material"):
            outline(ax[f"{kind} {r}"], m["true_fg"])
    fig.colorbar(im, ax=[ax[f"material {r}"] for r in range(len(MAP_ROWS))], fraction=0.05, pad=0.02)
    fig.suptitle(exp_label, fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_side_by_side(labels, all_maps, fits_df, path):
    """Every experiment in a row: the truth, the labels with the scale set by hand, the labels with NO scale
    chosen, and the share of models that call a pixel an object in that last case."""
    import matplotlib.pyplot as plt
    materials, regions = style()
    n = len(labels)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.55 * n), layout="constrained")
    axes = np.atleast_2d(axes)
    hand, none = MAP_ROWS[0][0], MAP_ROWS[1][0]
    for r, (name, label) in enumerate(labels):
        mh, mn = all_maps[(name, hand)], all_maps[(name, none)]
        show(axes[r, 0], mh["regions"], "true materials (object.fits)" if r == 0 else "", cmap=regions, vmin=0, vmax=5)
        iou = lambda s: fits_df[(fits_df.experiment == name) & (fits_df.setting == s)].outline_iou.iloc[0]  # noqa: E731
        show(axes[r, 1], mh["labels"], "labels: satellite max = 1 by hand" if r == 0 else "", cmap=materials, vmin=0, vmax=N_CLASSES - 1)
        show(axes[r, 2], mn["labels"], "labels: NO scale chosen" if r == 0 else "", cmap=materials, vmin=0, vmax=N_CLASSES - 1)
        show(axes[r, 3], mn["objectness"], "share of models: an object (no scale)" if r == 0 else "", cmap="Blues", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(label, fontsize=10, fontweight="bold")
        axes[r, 1].set_xlabel(f"IoU {iou(hand):.2f}", fontsize=8.5)
        axes[r, 2].set_xlabel(f"IoU {iou(none):.2f}", fontsize=8.5)
        for c, m in ((1, mh), (2, mn), (3, mn)):
            outline(axes[r, c], m["true_fg"])
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------- the table
def summary_table(exps, syn, rob, fits):
    """One row per experiment; (higher) / (lower) says which way is better."""
    def syn_at(name, stage, lg, metric="material_acc"):
        part = syn[(syn.experiment == name) & (syn.scenes == "satellite") & (syn.stage == stage) & (syn.log10_gain == lg)]
        return part[metric].mean()

    def rob_at(name, test, level, metric):
        part = rob[(rob.experiment == name) & (rob.test == test) & np.isclose(rob.level, level)]
        return part[metric].mean()

    def fit_at(name, setting, column):
        part = fits[(fits.experiment == name) & (fits.setting == setting)]
        return part[column].iloc[0] if len(part) else np.nan

    cols = {
        "synthetic: material acc at the trained brightness (higher)": lambda n: syn_at(n, "after_noise", 0.0),
        "synthetic: material acc, whole image 10x brighter (higher)": lambda n: syn_at(n, "after_noise", 1.0),
        "synthetic: material acc, whole image 10x dimmer (higher)": lambda n: syn_at(n, "after_noise", -1.0),
        "synthetic: material acc, object 1/3 as bright, same noise (higher)": lambda n: syn_at(n, "before_noise", -0.5),
        "synthetic: material acc, object 1/10 as bright, same noise (higher)": lambda n: syn_at(n, "before_noise", -1.0),
        "band curve sd 0.2: material acc (higher)": lambda n: rob_at(n, "band curve", 0.2, "material_acc"),
        "halo at 1% of the object: background given a material (lower)": lambda n: rob_at(n, "halo", -2.0, "false_fg_share"),
        "restoration-like noise sd 0.03: background given a material (lower)": lambda n: rob_at(n, "artifacts", 0.03, "false_fg_share"),
        "empty scene, training noise: pixels given a material (lower)": lambda n: rob_at(n, "empty scene", 1.0, "labelled_share"),
        "empty scene, 1/100 of that noise: pixels given a material (lower)": lambda n: rob_at(n, "empty scene", 0.01, "labelled_share"),
        "estimate.fits, NO scale chosen: outline IoU (higher)": lambda n: fit_at(n, "no scale chosen", "outline_iou"),
        "estimate.fits, NO scale chosen: materials ARI (higher)": lambda n: fit_at(n, "no scale chosen", "ari_materials"),
        "estimate.fits, NO scale chosen: false pixels (lower)": lambda n: fit_at(n, "no scale chosen", "false_px"),
        "estimate.fits, by hand (max = 1): outline IoU (higher)": lambda n: fit_at(n, "satellite max = 1 by hand", "outline_iou"),
        "estimate.fits, by hand (max = 1): materials ARI (higher)": lambda n: fit_at(n, "satellite max = 1 by hand", "ari_materials"),
        "estimate.fits, blank window: pixels given a material (lower)": lambda n: fit_at(n, "blank window, own scale", "labelled_px"),
    }
    rows = [{"experiment": e.label, **{c: f(e.name) for c, f in cols.items()}} for e in exps]
    return pd.DataFrame(rows).set_index("experiment")


def markdown(table):
    def fmt(v):
        return "" if pd.isna(v) else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.3f}")
    delta = table - table.iloc[0]                       # the first row is the baseline
    lines = ["# Comparison with the original model", "",
             "Mean over each experiment's runs. (higher) / (lower) says which way is better. "
             "The change column is new minus original.", ""]
    for col in table.columns:
        lines += [f"## {col}", "", "| experiment | value | change from original |", "|---|---|---|"]
        for name in table.index:
            d = "" if name == table.index[0] else ("%+.3f" % delta.loc[name, col] if abs(delta.loc[name, col]) < 100 else "%+.0f" % delta.loc[name, col])
            lines.append(f"| {name} | {fmt(table.loc[name, col])} | {d} |")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="estimate_10band", help="the previous model, compared against (default estimate_10band)")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="the new ones (folders under --logdir); default: every preset folder found there")
    ap.add_argument("--logdir", default=str(ROOT / "logs"))
    ap.add_argument("--max-runs", type=int, default=None, help="use only the first N runs of each experiment")
    ap.add_argument("--n-scenes", type=int, default=200, help="synthetic scenes per kind and level")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--skip-synthetic", action="store_true", help="only estimate.fits (the maps and their scores)")
    ap.add_argument("--out", default=str(ROOT / "outputs/brightness_compare"))
    args = ap.parse_args()

    logdir = Path(args.logdir)
    names = args.experiments
    if names is None:
        names = [experiment_name(n) for n in ORDER if n != "base" and (logdir / experiment_name(n) / "config.yaml").is_file()]
        if not names:
            sys.exit(f"no experiment folders of the known variants under {logdir}; run slurm/train_brightness.sh first "
                     f"(variants: {', '.join(n for n in PRESETS if n != 'base')})")
    names = [args.baseline] + [n for n in names if n != args.baseline]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    exps = [ev.Experiment(logdir, n, args.max_runs) for n in names]
    for e in exps:
        print(f"{e.label:14s} {len(e.models):2d} runs   input_norm={e.norm:9s} "
              f"gain={e.config.data.params.sim.get('gain_log10_range', None)} "
              f"curve={e.config.data.params.sim.get('response_log10_std', 0.0)} "
              f"empty_prob={e.config.data.params.sim.get('empty_prob', 0.0)}")

    if args.skip_synthetic:
        syn = rob = pd.DataFrame(columns=["experiment", "scenes", "stage", "log10_gain", "test", "level"])
    else:
        syn, rob = ev.run_synthetic(exps, args, out)

    print("estimate.fits against object.fits")
    scene = ev.FitsScene(exps[0].data.cfg.wavelengths_nm)
    fits, maps = ev.fits_results(exps, scene)
    fits.to_csv(out / "estimate_fits.csv", index=False)

    print("drawing the maps")
    for e in exps:
        plot_maps(e.label, {k: maps[(e.name, k)] for k, _ in MAP_ROWS}, out / f"maps_{e.label}.png")
    plot_side_by_side([(e.name, e.label) for e in exps], maps, fits, out / "maps_side_by_side.png")

    table = summary_table(exps, syn, rob, fits)
    table.round(4).to_csv(out / "comparison_summary.csv")
    (out / "comparison_summary.md").write_text(markdown(table))
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60):
        for chunk in (table.iloc[:, :6], table.iloc[:, 6:10], table.iloc[:, 10:]):
            print(chunk.round(3).to_string(), "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
