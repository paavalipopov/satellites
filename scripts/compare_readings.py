"""estimate.fits read two ways, same models: does the collaborators' recipe change what the models find?

For each experiment it scores estimate.fits against object.fits once per reading (scripts/fits_reading.py)
and puts the results next to each other: a table (outline IoU, materials ARI, false pixels, ...) and
a figure of the maps. The models and their training are untouched; only the numbers fed to them change.

    python scripts/compare_readings.py                                    # original + curve_max + empty_max, solar vs radiometry
    python scripts/compare_readings.py --experiments estimate_10band estimate_10band_full_max \\
        --readings solar radiometry radiometry:1

A reading is `solar` (notebook 06, the previous results), `radiometry` (the collaborators' recipe, no sun),
or `radiometry_sun`; add `:mag` to set the overall constant of a radiometry reading, e.g. `radiometry:1`
(the collaborators' own default of mag 1 gives values 100x SMALLER than the default mag 6: a brighter source means a bigger divisor).

Read the "by hand" rows first: there the satellite's maximum is set to 1, so only the COLOURS change between
readings. The "no scale chosen" rows also change with the overall constant, which the recipe leaves free.

Writes to --out (default outputs/brightness_readings): reading_comparison.csv / .md, readings_side_by_side.png.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import brightness_compare as bc  # noqa: E402
import brightness_eval as ev  # noqa: E402
from brightness_presets import experiment_name  # noqa: E402

SETTINGS = ["satellite max = 1 by hand", "no scale chosen", "no scale chosen + cutoff",
            "companion on its own scale by hand", "blank window, own scale"]
METRICS = ["outline_iou", "ari_materials", "false_px", "labelled_px", "satellite_iou", "companion_px_found"]


def parse(spec):
    name, _, mag = spec.partition(":")
    return name, (float(mag) if mag else None), spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="folders under --logdir (default: estimate_10band, curve_max and empty_max)")
    ap.add_argument("--readings", nargs="+", default=["solar", "radiometry"])
    ap.add_argument("--logdir", default=str(ROOT / "logs"))
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs/brightness_readings"))
    args = ap.parse_args()

    names = args.experiments or ["estimate_10band", experiment_name("curve_max"), experiment_name("empty_max")]
    logdir = Path(args.logdir)
    missing = [n for n in names if not (logdir / n / "config.yaml").is_file()]
    if missing:
        sys.exit(f"no experiment folder under {logdir} for: {', '.join(missing)}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    exps = [ev.Experiment(logdir, n, args.max_runs) for n in names]
    lam = exps[0].data.cfg.wavelengths_nm

    frames, maps, labels, facts = [], {}, [], []
    for spec in args.readings:
        reading, mag, tag = parse(spec)
        scene = ev.FitsScene(lam, reading, mag)
        facts.append((tag, float(scene.sat_max), float(scene.comp_max)))
        df, m = ev.fits_results(exps, scene)
        df["reading"] = tag
        df["key"] = df.experiment + " [" + tag + "]"
        frames.append(df)
        for (name, setting), v in m.items():
            maps[(f"{name} [{tag}]", setting)] = v
    fits = pd.concat(frames, ignore_index=True)
    fits.to_csv(out / "reading_comparison.csv", index=False)

    # the figure: one row per experiment and reading
    plot_df = fits.assign(experiment=fits.key)
    for e in exps:
        for spec in args.readings:
            labels.append((f"{e.name} [{spec}]", f"{e.label}\n{spec}"))
    bc.plot_side_by_side(labels, maps, plot_df, out / "readings_side_by_side.png")

    # the table
    short = {e.name: e.label for e in exps}
    lines = ["# estimate.fits: the same models, read two ways", "",
             "Readings: " + ", ".join(f"`{s}`" for s in args.readings) + ". `solar` = notebook 06, the previous results; "
             "`radiometry` = the collaborators' recipe (scripts/fits_reading.py). Only the numbers fed to the models change.", "",
             "Brightest value of the satellite / of the companion after the reading (the overall constant is free for the "
             "radiometry readings):", ""]
    lines += [f"- `{t}`: {s:.4g} / {c:.4g}" for t, s, c in facts] + [""]
    for setting in SETTINGS:
        part = fits[fits.setting == setting]
        if part.empty:
            continue
        lines += [f"## {setting}", "", "| experiment | reading | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
        for e in exps:
            for spec in args.readings:
                r = part[(part.experiment == e.name) & (part.reading == spec)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(f"| {short[e.name]} | {spec} | " + " | ".join(
                    "" if pd.isna(r[m]) else (f"{r[m]:.0f}" if m.endswith("px") or m.endswith("found") else f"{r[m]:.2f}")
                    for m in METRICS) + " |")
        lines.append("")
    (out / "reading_comparison.md").write_text("\n".join(lines))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for setting in SETTINGS[:2]:
            print(f"\n{setting}")
            print(fits[fits.setting == setting].assign(model=lambda d: d.experiment.map(short))
                  .set_index(["model", "reading"])[METRICS].round(2).to_string())
    print("\nwrote", out)


if __name__ == "__main__":
    main()
