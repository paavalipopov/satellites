"""The whole pipeline on estimate.fits: how each choice changes the shape and the materials found.

Choices at test time (no training; every combination is scored):
  scaling   none | single | single_mean | single_p99 | per_object | per_object_mean | per_object_p99 | per_object_faint   scripts/auto_scale.py
  outline   model | mask@0.01 | fill@0.01 | mask@0.001 | fill@0.001 | mask@0.01+faint | fill@0.01+faint   scripts/postprocess.py
  cleanup   off | tidy | tidy+smooth                          scripts/postprocess.py
  vote      0.3 | 0.5 | 0.7   (the share of runs that must call a pixel an object)

Choices at training time are the experiments themselves (folders under --logdir; each is a set of runs, 3 in
slurm/run_pipeline.sh): by default every variant of scripts/brightness_presets.py (the original, the earlier rounds such as
aug_norm, gain_flux and full_p99, the 24 pair_* variants, fg3, fg10, fg3_dice, lightnoise, bank). An experiment without a
folder is skipped with a message. Use --max-runs 3 so that every experiment is scored on the same 3 seeds (runs 0-2).

Every cell reports (window of 41 x 41 px round satellite and companion, as notebook 06):
  outline_iou      overlap of everything labelled with the true object (1 = perfect)
  ari_materials    1 when the labels split the object's pixels into materials like the truth, 0 at chance
  false_px         labelled pixels that are not object (lower is better)
  satellite_px_found / companion_px_found   true pixels labelled (the true companion is 1 pixel, 18 px right of the satellite)
  faint_found      share of the FAINT true pixels (estimate dimmer than 5% of its brightest) that were labelled:
                   the faint solar panels that models keep dropping
and, for the vote rows, the same with every run on its own ("each run"): the mean over runs and the spread (std).

    python scripts/compare_pipeline.py
    python scripts/compare_pipeline.py --experiments estimate_10band estimate_10band_fg3 --readings radiometry

Writes to --out (default outputs/pipeline_compare): pipeline_grid.csv (every cell), pipeline_summary.md,
pipeline_maps_<reading>.png. The by-hand rows (a person chose the objects and the scales) are the reference.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import auto_scale  # noqa: E402
import brightness_eval as ev  # noqa: E402
import compare_scaling as cs  # noqa: E402
import postprocess as pp  # noqa: E402
from brightness_presets import PRESETS, experiment_name  # noqa: E402

DEFAULT_EXPERIMENTS = [experiment_name(n) for n in PRESETS]      # every variant there is; the ones not trained are skipped
TODAY = dict(scaling="none", outline="model", cleanup="off", vote="0.5")       # what a model gets with no extra step
FAINT = 0.05
METRICS = ["outline_iou", "ari_materials", "false_px", "satellite_px_found", "companion_px_found", "faint_found"]
CHOICES = ["scaling", "outline", "cleanup", "vote"]


def metrics(scene, labels, size, centre):
    """The scores of one labelled window (cs.window_row) plus the faint-pixel share."""
    row = cs.window_row(scene, labels, None, None, size, centre)
    ts = scene.win(scene.true_sat, size, centre)
    row["satellite_px_found"] = int(((labels > 0) & ts).sum())
    faint = scene.win(scene.true_fg, size, centre) & (scene.win(scene.total, size, centre) < FAINT * scene.total.max())
    row["faint_px_true"] = int(faint.sum())
    row["faint_found"] = float(((labels > 0) & faint).sum() / faint.sum()) if faint.any() else np.nan
    return row


def grid_rows(scene, e, reading, size, centre, cube, preds_by_scaling, outlines=pp.OUTLINES, cleanups=pp.CLEANUPS, votes=pp.VOTES):
    """Every combination of the test-time choices asked for, for one experiment."""
    rows = []
    for scaling, outline, clean in itertools.product(preds_by_scaling, outlines, cleanups):
        preds = preds_by_scaling[scaling]
        base = {"reading": reading, "experiment": e.name, "window": "both objects",
                "scaling": scaling, "outline": outline, "cleanup": clean}
        for v in votes:
            labels = pp.finish(pp.vote(preds, v), cube, outline, clean)
            rows.append({**base, "vote": str(v), **metrics(scene, labels, size, centre)})
        per_run = pd.DataFrame([metrics(scene, pp.finish(p, cube, outline, clean), size, centre) for p in preds])
        agg = per_run.mean().to_dict()
        agg["outline_iou_std"], agg["ari_materials_std"] = float(per_run.outline_iou.std(ddof=0)), float(per_run.ari_materials.std(ddof=0))
        rows.append({**base, "vote": "each run", **agg})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=None, help="folders under --logdir")
    ap.add_argument("--logdir", default=str(ROOT / "logs"))
    ap.add_argument("--readings", nargs="+", default=["solar", "radiometry"], choices=("solar", "radiometry", "radiometry_sun"))
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs/pipeline_compare"))
    ap.add_argument("--scalings", nargs="+", default=list(auto_scale.MODES), choices=auto_scale.MODES,
                    help="test-time scalings to try (default: all)")
    ap.add_argument("--outlines", nargs="+", default=list(pp.OUTLINES), choices=pp.OUTLINES, help="default: all")
    ap.add_argument("--cleanups", nargs="+", default=list(pp.CLEANUPS), choices=pp.CLEANUPS, help="default: all")
    ap.add_argument("--votes", nargs="+", type=float, default=list(pp.VOTES), help="default: 0.3 0.5 0.7")
    ap.add_argument("--title", default="The pipeline on estimate.fits", help="heading of the summary")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    names = args.experiments or DEFAULT_EXPERIMENTS
    have = [n for n in names if (logdir / n / "config.yaml").is_file()]
    for n in names:
        if n not in have:
            print(f"skipping {n}: no folder with a config.yaml under {logdir}", flush=True)
    if not have:
        sys.exit(f"none of the experiments exist under {logdir}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exps = [ev.Experiment(logdir, n, args.max_runs) for n in have]
    for e in exps:
        print(f"{e.name}: {len(e.models)} runs, input_norm={e.norm}", flush=True)
    lam = exps[0].data.cfg.wavelengths_nm

    frames = []
    for reading in args.readings:
        scene = ev.FitsScene(lam, reading)
        size, centre = scene.SIZE, scene.centre
        cube = scene.win(scene.reading, size, centre).copy()
        blank = scene.win(scene.reading, size, scene.blank_centre).copy()
        rows, best_maps = [], {}
        for e in exps:
            print(f"  {reading}: {e.name}", flush=True)
            preds = {m: cs.run_mode(e, cube, m) for m in args.scalings}
            erows = grid_rows(scene, e, reading, size, centre, cube, preds, args.outlines, args.cleanups, args.votes)
            rows += erows
            for m in args.scalings:                         # nothing in the window: every labelled pixel is a false one
                lab = pp.vote(cs.run_mode(e, blank, m), 0.5)
                rows.append({"reading": reading, "experiment": e.name, "window": "blank window", "scaling": m, "outline": "model",
                             "cleanup": "off", "vote": "0.5", "false_px": int((lab > 0).sum()), "labelled_px": int((lab > 0).sum())})
            if "none" in preds:
                best_maps[(e.name, "today")] = pp.vote(preds["none"], 0.5)
            if "single" in preds:
                best_maps[(e.name, "auto: single")] = pp.vote(preds["single"], 0.5)
            voted = [r for r in erows if r["vote"] != "each run"]
            best = max(voted, key=lambda r: r["outline_iou"] + (0.0 if np.isnan(r["ari_materials"]) else r["ari_materials"]))
            best_maps[(e.name, "best setting")] = pp.finish(pp.vote(preds[best["scaling"]], float(best["vote"])), cube,
                                                            best["outline"], best["cleanup"])
        by_hand, hand_maps = ev.fits_results(exps, scene)
        for _, r in by_hand[by_hand.setting == "satellite max = 1 by hand"].iterrows():
            rows.append({**r.to_dict(), "reading": reading, "scaling": "by hand", "outline": "model", "cleanup": "off",
                         "vote": "0.5", "window": "both objects"})
            best_maps[(r.experiment, "by hand")] = hand_maps[(r.experiment, r.setting)]["labels"]
        frames.append(pd.DataFrame(rows))
        plot_maps(scene, exps, best_maps, out / f"pipeline_maps_{reading}.png", reading)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "pipeline_grid.csv", index=False)
    write_summary(out, df, exps, args.title)
    print("wrote", out / "pipeline_summary.md")


def plot_maps(scene, exps, maps, path, reading):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data_generation import N_CLASSES

    titles = ["truth", "today", "auto: single", "by hand", "best setting"]
    fig, axes = plt.subplots(len(exps), len(titles), figsize=(2.3 * len(titles), 2.4 * len(exps)), squeeze=False)
    cmap = plt.get_cmap("nipy_spectral").copy()
    cmap.set_bad("white")
    truth = scene.win(scene.regions, scene.SIZE, scene.centre)
    for i, e in enumerate(exps):
        for j, title in enumerate(titles):
            ax = axes[i][j]
            ax.axis("off")
            lab = truth if title == "truth" else maps.get((e.name, title))
            if lab is None:
                continue
            ax.imshow(np.ma.masked_equal(lab, 0), cmap=cmap, vmin=1, vmax=N_CLASSES, interpolation="nearest")
            if i == 0:
                ax.set_title(title, fontsize=7)
            if j == 0:
                ax.text(-0.05, 0.5, e.label, transform=ax.transAxes, ha="right", va="center", fontsize=7, rotation=90)
    fig.suptitle(f"estimate.fits, reading '{reading}': material labels (white = background)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary(out, df, exps, title="The pipeline on estimate.fits"):
    short = {e.name: e.label for e in exps}
    both = df[df.window == "both objects"].copy()
    voted = both[(both.vote != "each run") & (both.scaling != "by hand")]
    md = [f"# {title}", "",
          "Models: " + ", ".join(f"`{e.label}`" for e in exps) + f" ({len(exps[0].models)} runs in the first). "
          "Choices: see `python scripts/compare_pipeline.py --help`. Higher is better for iou, ari, found and faint_found; lower for false_px. "
          "The true companion is 1 pixel, 18 px right of the satellite (companion_px_found is 0 or 1). By hand = the notebook's reference (a person chose the object and its scale).", ""]
    for reading in both.reading.unique():
        rd, rv = both[both.reading == reading], voted[voted.reading == reading]
        md += [f"## Reading `{reading}`", ""]
        # 1. today / by hand / each model's best setting
        rows = []
        for name in rd.experiment.unique():
            part = rv[rv.experiment == name]
            plain = (part.outline == "model") & (part.cleanup == "off") & (part.vote == "0.5")
            today = part[plain & (part.scaling == "none")]
            auto = part[plain & (part.scaling == "single")]
            hand = rd[(rd.experiment == name) & (rd.scaling == "by hand")]
            part = part.assign(score=part.outline_iou + part.ari_materials.fillna(0))
            best = part.sort_values("score", ascending=False).head(1)
            for label, sub in (("today (no step)", today), ("automatic max scaling only", auto), ("by hand", hand), ("best setting", best)):
                for _, r in sub.iterrows():
                    setting = "" if label != "best setting" else f"{r.scaling} / {r.outline} / {r.cleanup} / vote {r.vote}"
                    rows.append({"model": short[name], "row": label, "setting": setting, **{m: r.get(m, np.nan) for m in METRICS}})
        md += ["### Each model: no step, automatic max scaling, by hand, and its best setting (best = highest iou + ari; a heuristic, look at the other columns)", "",
               cs.md_table(pd.DataFrame(rows).set_index(["model", "row"]).round(2)), ""]
        # 2. what each choice does, averaged over everything else
        md += ["### What each choice does (mean over all models and all other choices)", ""]
        for choice in CHOICES:
            if rv[choice].nunique() < 2:
                continue
            t = rv.groupby(choice)[METRICS].mean().round(2)
            md += [f"`{choice}`:", "", cs.md_table(t), ""]
        # 2b. every model with every scaling (the other choices at their plain setting, if they were tried)
        plain = rv[(rv.outline == "model") & (rv.cleanup == "off") & (rv.vote == "0.5")]
        if plain.scaling.nunique() > 1:
            md += ["### Every model with every scaling (model outline, no cleanup, vote 0.5)", ""]
            for m in ("outline_iou", "ari_materials", "faint_found", "companion_px_found"):
                t = plain.assign(model=plain.experiment.map(short)).pivot_table(index="model", columns="scaling", values=m).round(2)
                md += [f"`{m}`:", "", cs.md_table(t), ""]
        # 3. spread over runs, for the model-only / single-scale / best-looking settings
        each = rd[rd.vote == "each run"]
        sel = each[(each.outline == "model") & (each.cleanup == "off") & (each.scaling.str.startswith("single") | (each.scaling == "none"))]
        t = sel.assign(model=sel.experiment.map(short)).set_index(["model", "scaling"])[["outline_iou", "outline_iou_std", "ari_materials", "ari_materials_std"]].round(2)
        md += ["### Every run on its own (mean and std over the runs; no voting)", "", cs.md_table(t), ""]
        # 4. nothing in the window
        blank = df[(df.window == "blank window") & (df.reading == reading)]
        md += ["### A window with nothing in it: labelled pixels (all false)", "",
               cs.md_table(blank.assign(model=blank.experiment.map(short)).pivot_table(index="model", columns="scaling", values="false_px").round(0)), ""]
    (out / "pipeline_summary.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
