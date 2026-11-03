"""Does a model cope with brightness, with the response of the system, and with empty sky, without hand-set scales?

Scores trained experiments (each a folder of runs under logs/) on:

1. Synthetic satellite scenes at many brightness levels. The SAME scenes and noise are rescaled by a
   gain, either before the sensor noise (a dimmer object against the same noise: low signal-to-noise)
   or after it (a change of units: the signal-to-noise ratio is unchanged).
2. Robustness tests, on the same scenes: a smooth random curve over the bands (what sunlight,
   atmosphere, optics and the detector do to a real spectrum), scenes with NO object at several noise
   levels, a faint halo around the object, and restoration-like artifacts. Material accuracy, and the
   share of background pixels wrongly given a material.
3. assets/new_input/estimate.fits against object.fits, read like notebook 06 (V / (E.dlam)): with the
   satellite's max set to 1 by hand (notebook 06's best case), with NO scale chosen, with the
   companion on its own scale, on a window with nothing in it, and so on.

Run from the project root:

    python scripts/brightness_eval.py --experiments estimate_10band estimate_10band_aug_norm ...

To compare against the original model and draw the notebook-06 maps, use scripts/brightness_compare.py
(it calls this). Writes tables (csv) and figures (png) to --out (default outputs/brightness_eval).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_generation import N_CLASSES, SpectralLibraryWarning, generate_sample  # noqa: E402
from src.datasets.hsi import HSIDataset, scene_seed  # noqa: E402
from src.registry import build_model  # noqa: E402

warnings.simplefilter("ignore", SpectralLibraryWarning)

BEFORE_NOISE = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]     # log10 gain: a dimmer object, same noise
AFTER_NOISE = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]   # log10 gain: the whole image rescaled

# robustness tests: name -> (x label, levels, scene kind, what the level changes)
ROBUSTNESS = {
    "band curve": ("sd of log10 of the curve over the bands", [0.0, 0.05, 0.1, 0.2, 0.4], "satellite"),
    "empty scene": ("noise, in units of the training sensor noise", [0.0, 0.01, 0.1, 1.0, 3.0], "empty"),
    "halo": ("log10 of halo peak / object peak", [-4.0, -3.0, -2.0, -1.5, -1.0], "satellite"),
    "artifacts": ("sd of the artifacts (training noise: 0.02)", [0.002, 0.01, 0.03, 0.1], "satellite"),
}


def short(name):
    """estimate_10band -> original; estimate_10band_aug_norm -> aug_norm."""
    return "original" if name == "estimate_10band" else name.replace("estimate_10band_", "")


def clean_cfg(cfg):
    """The experiment's generator settings with every training-time option switched off:
    what a test scene should look like, whatever the model was trained on."""
    return cfg.with_(gain_log10_range=None, gain_stage="before_noise", response_log10_std=0.0,
                     empty_prob=0.0, halo_log10_range=None, artifact_std_range=None, noise_bank=None)


# ----------------------------------------------------------------------------- loading
class Experiment:
    """The runs of one experiment, loaded from its logged config and best checkpoints."""

    def __init__(self, logdir, name, max_runs=None):
        self.name = name
        folder = Path(logdir) / name
        self.config = OmegaConf.load(folder / "config.yaml")
        runs = sorted(folder.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
        runs = [r for r in runs if (r / "model.best.pth").is_file()][:max_runs]
        if not runs:
            raise FileNotFoundError(f"no run_*/model.best.pth under {folder}")
        self.models = []
        for run in runs:
            model = build_model(self.config)  # input_norm etc. come from the logged config
            model.load_state_dict(torch.load(run / "model.best.pth", map_location="cpu"))
            self.models.append(model.eval())
        self.data = HSIDataset(self.config.data.params)
        self.norm = self.config.model.params.get("input_norm", "none")
        self.normalizes = self.norm != "none"

    @property
    def label(self):
        return short(self.name)

    @property
    def external_scale_ok(self):
        """Whether the scale can be taken from a bigger image (a single number per input)."""
        return self.norm == "max"

    def predict(self, x, scale_from=None):
        """x: [N, bands, H, W] float32 -> labels [runs, N, H, W].
        scale_from: a whole image [1, bands, H', W']; the input is divided by the number the model's
        own normalization takes from THAT image (only for experiments with external_scale_ok)."""
        with torch.no_grad():
            outs = []
            for m in self.models:
                scale = m.input_scale(scale_from) if scale_from is not None else None
                outs.append((m(x) if scale is None else m(x, scale=scale)).argmax(1))
            return torch.stack(outs).numpy()


# ----------------------------------------------------------------------------- synthetic
def score_samples(experiments, samples, extra):
    """One row per experiment and run, on a list of Samples."""
    x = torch.from_numpy(np.stack([s.cube for s in samples])).float()
    mask = np.stack([s.mask for s in samples])
    fg = mask > 0
    rows = []
    for e in experiments:
        for r, p in enumerate(e.predict(x)):
            found = p > 0
            rows.append({"experiment": e.name, "run": r, **extra,
                         "material_acc": (p == mask)[fg].mean() if fg.any() else np.nan,
                         "foreground_iou": (found & fg).sum() / max((found | fg).sum(), 1) if fg.any() else np.nan,
                         "false_fg_share": found[~fg].mean(),
                         "labelled_share": found.mean()})
    return rows


def check_same_bands(experiments):
    ref = experiments[0].data
    for e in experiments:
        if not np.allclose(e.data.cfg.wavelengths_nm, ref.cfg.wavelengths_nm):
            raise ValueError(f"{e.name} was trained on other bands than {experiments[0].name}")
    return ref


def synthetic_scores(experiments, n_scenes, seed):
    """Material accuracy etc. against the overall brightness of the scene."""
    ref = check_same_bands(experiments)
    rows = []
    for kind in ("satellite", "random"):
        for stage, gains in (("before_noise", BEFORE_NOISE), ("after_noise", AFTER_NOISE)):
            cfg = clean_cfg(ref.cfg).with_(gain_stage=stage)
            for lg in gains:
                samples = [generate_sample(cfg, scene_seed(seed, i), ref.library, scene=kind, gain=10.0 ** lg)
                           for i in range(n_scenes)]
                rows += score_samples(experiments, samples, {"scenes": kind, "stage": stage, "log10_gain": lg})
                print(f"  {kind:9s} {stage:12s} log10 gain {lg:+.1f} done", flush=True)
    return pd.DataFrame(rows)


def robustness_scores(experiments, n_scenes, seed):
    """The band-curve, empty-scene, halo and artifact tests: one row per test, level, experiment and run."""
    ref = check_same_bands(experiments)
    base = clean_cfg(ref.cfg)
    sigma = ref.cfg.sensor_noise_std
    rows = []
    for test, (_, levels, kind) in ROBUSTNESS.items():
        for v in levels:
            if test == "band curve":
                cfg = base.with_(response_log10_std=v)
            elif test == "empty scene":                    # no object; noise at a fixed multiple of the sensor noise
                cfg = base.with_(sensor_noise_std=sigma * v, empty_noise_log10_range=(0.0, 0.0))
            elif test == "halo":
                cfg = base.with_(halo_log10_range=(v, v))
            else:
                cfg = base.with_(artifact_std_range=(v, v))
            samples = [generate_sample(cfg, scene_seed(seed, 1, i), ref.library, scene=kind) for i in range(n_scenes)]
            rows += score_samples(experiments, samples, {"test": test, "level": v})
            print(f"  {test:12s} level {v:g} done", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plots
def colours(names):
    import matplotlib
    palette = matplotlib.colormaps["tab20"].colors
    out, k = {}, 0
    for n in names:
        if n == "estimate_10band":
            out[n] = "#222222"
        else:
            out[n] = palette[k % 20]; k += 1
    return out


def plot_synthetic(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = colours(sorted(df.experiment.unique(), key=lambda n: (n != "estimate_10band", n)))
    metrics = [("material_acc", "material accuracy (true foreground)"), ("foreground_iou", "foreground IoU")]
    stages = [("before_noise", "dimmer object, same sensor noise"), ("after_noise", "whole image rescaled")]
    fig, axes = plt.subplots(len(metrics), len(stages), figsize=(11, 6.8), sharey="row")
    for r, (metric, mname) in enumerate(metrics):
        for c, (stage, sname) in enumerate(stages):
            ax = axes[r, c]
            for name, part in df[(df.scenes == "satellite") & (df.stage == stage)].groupby("experiment"):
                g = part.groupby("log10_gain")[metric].agg(["mean", "std"]).fillna(0)
                ax.plot(g.index, g["mean"], marker="o", ms=3.5, color=col[name], label=short(name),
                        lw=2.4 if name == "estimate_10band" else 1.5)
                ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=col[name], alpha=0.1)
            ax.set_title(sname if r == 0 else "", fontsize=10)
            ax.set_xlabel("log10 gain (0 = as trained)" if r == len(metrics) - 1 else "")
            ax.set_ylabel(mname if c == 0 else "")
            ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Satellite scenes at different brightness (mean and sd over runs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_robustness(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = colours(sorted(df.experiment.unique(), key=lambda n: (n != "estimate_10band", n)))
    panels = [("band curve", "material_acc", "material accuracy: a smooth curve over the bands"),
              ("halo", "false_fg_share", "background given a material: faint halo around the object"),
              ("artifacts", "false_fg_share", "background given a material: restoration-like noise"),
              ("empty scene", "labelled_share", "empty scene: pixels given a material")]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    for ax, (test, metric, title) in zip(axes.ravel(), panels):
        for name, part in df[df.test == test].groupby("experiment"):
            g = part.groupby("level")[metric].mean()
            ax.plot(range(len(g)), g.values, marker="o", ms=4, color=col[name], label=short(name),
                    lw=2.4 if name == "estimate_10band" else 1.5)
        levels = ROBUSTNESS[test][1]
        ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"{v:g}" for v in levels])
        ax.set_xlabel(ROBUSTNESS[test][0], fontsize=8.5); ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 1.02 if metric == "material_acc" else None); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Robustness (mean over runs): higher is better in the top left, lower elsewhere", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ----------------------------------------------------------------------------- estimate.fits
def ari(truth_ids, pred):
    """Adjusted Rand index (as notebook 03): 1 when the labels split the pixels like the truth, 0 at chance."""
    n = pd.crosstab(truth_ids, pred).to_numpy()
    pairs = lambda counts: (counts * (counts - 1) / 2).sum()  # noqa: E731
    rows, cols = pairs(n.sum(1)), pairs(n.sum(0))
    expected = rows * cols / pairs(n.sum(keepdims=True))
    denom = (rows + cols) / 2 - expected
    return float((pairs(n) - expected) / denom) if denom else np.nan


def combine(preds):
    """As notebook 06: an object where most runs give a material, with the material most runs give;
    also the share of runs that call each pixel an object, and that give it that material."""
    counts = (preds[..., None] == np.arange(1, N_CLASSES)).sum(0)
    objectness = (preds > 0).mean(0)
    labels = np.where(objectness >= 0.5, counts.argmax(-1) + 1, 0)
    return labels, objectness, counts.max(-1) / len(preds)


class FitsScene:
    """assets/new_input/estimate.fits and object.fits, read like notebook 06 (V / (E.dlam), blue to red)."""

    SIZE = 41          # px, the patch around both objects (notebook 06)
    COMPANION = 21     # px, the window around the companion alone

    def __init__(self, lam):
        import pvlib
        from astropy.io import fits
        from scipy import ndimage

        solar = pvlib.spectrum.get_reference_spectra(standard="ASTM G173-03")["extraterrestrial"]
        divisor = np.interp(lam, solar.index, solar.values) * lam ** 2 / 6400          # E * d_lambda
        cube = fits.getdata(ROOT / "assets/new_input/estimate.fits").astype(float)[::-1]    # blue -> red
        truth = fits.getdata(ROOT / "assets/new_input/object.fits").astype(float)[::-1]
        self.reading = cube / divisor[:, None, None]
        self.total = cube.sum(0)
        total = self.total

        groups, n = ndimage.label(total > 0, structure=np.ones((3, 3)))
        self.satellite = groups == groups[np.unravel_index(total.argmax(), total.shape)]
        totals = ndimage.sum(total, groups, range(1, n + 1))
        totals[groups[self.satellite][0] - 1] = 0
        self.companion = groups == totals.argmax() + 1
        self.sat_px = self.satellite & (total > 0.01 * total.max())
        self.comp_px = self.companion & (total > 0.05 * total[self.companion].max())
        self.centre = tuple(int(round(v)) for v in (np.array(ndimage.center_of_mass(total * self.sat_px))
                                                    + ndimage.center_of_mass(total * self.comp_px)) / 2)
        self.comp_centre = tuple(int(round(v)) for v in ndimage.center_of_mass(total * self.comp_px))

        self.true_fg = truth.sum(0) > 0
        tgroups, tn = ndimage.label(self.true_fg, structure=np.ones((3, 3)))
        sizes = ndimage.sum(self.true_fg, tgroups, range(1, tn + 1))
        self.true_sat = tgroups == int(np.argmax(sizes)) + 1
        self.true_comp = self.true_fg & ~self.true_sat
        # the true materials A-E, as notebook 06: one spectrum per material, brightness removed
        spectra = (truth[:, self.true_fg] / truth.sum(0)[self.true_fg]).round(6).T
        _, first, ids = np.unique(spectra, axis=0, return_index=True, return_inverse=True)
        self.regions = np.zeros(self.true_fg.shape, int)
        self.regions[self.true_fg] = np.argsort(np.argsort(first))[ids.ravel()] + 1

        # brightest speck far from the objects: a window with (almost) nothing in it
        yy, xx = np.mgrid[:total.shape[0], :total.shape[1]]
        far = (yy - self.centre[0]) ** 2 + (xx - self.centre[1]) ** 2 > 100 ** 2
        y, x = np.unravel_index(np.where(far, total, -1).argmax(), total.shape)
        h = self.SIZE // 2
        self.blank_centre = (int(np.clip(y, h, total.shape[0] - h - 1)), int(np.clip(x, h, total.shape[1] - h - 1)))

        self.sat_max = self.reading[:, self.sat_px].max()
        self.comp_max = self.reading[:, self.comp_px].max()

    @staticmethod
    def win(a, size, centre):
        (y, x), h = centre, size // 2
        return a[..., y - h:y + h + 1, x - h:x + h + 1]

    def settings(self):
        """name -> dict(image, centre, size, external [take the scale from the whole image], cutoff)."""
        own = np.full(self.total.shape, 1.0 / self.sat_max)
        own[self.comp_px] = 1.0 / self.comp_max
        alone = np.where(self.satellite, 0.0, self.reading)
        S, C = self.SIZE, self.COMPANION
        return {
            "satellite max = 1 by hand": dict(image=self.reading / self.sat_max, centre=self.centre, size=S),
            "no scale chosen": dict(image=self.reading, centre=self.centre, size=S),
            "no scale chosen + cutoff": dict(image=self.reading, centre=self.centre, size=S, cutoff=0.01),
            "companion on its own scale by hand": dict(image=self.reading * own, centre=self.centre, size=S),
            "companion alone, no scale chosen": dict(image=alone, centre=self.comp_centre, size=C),
            "companion window, scale from whole image": dict(image=self.reading, centre=self.comp_centre, size=C, external=True),
            "blank window, own scale": dict(image=self.reading, centre=self.blank_centre, size=S),
            "blank window, scale from whole image": dict(image=self.reading, centre=self.blank_centre, size=S, external=True),
        }


def fits_results(experiments, scene=None):
    """Scores for every experiment and setting, and the maps behind them.
    Returns (DataFrame, {(experiment name, setting): dict of arrays})."""
    scene = scene or FitsScene(experiments[0].data.cfg.wavelengths_nm)
    whole = torch.from_numpy(scene.reading[None]).float()
    rows, maps = [], {}
    for e in experiments:
        for name, s in scene.settings().items():
            if s.get("external") and not e.external_scale_ok:
                continue                                    # the scale of these models is not one number
            c, size = s["centre"], s["size"]
            inp = torch.from_numpy(scene.win(s["image"], size, c).copy()).float()[None]
            preds = e.predict(inp, scale_from=whole if s.get("external") else None)[:, 0]    # [runs, H, W]
            labels, objectness, agreement = combine(preds)
            variants = [(name, labels)]
            if s.get("cutoff"):                              # drop pixels below a fraction of the brightest
                keep = scene.win(scene.total >= s["cutoff"] * scene.total.max(), size, c)
                variants = [(name, np.where(keep, labels, 0))]
            for label_name, lab in variants:
                found = lab > 0
                tf, tr = scene.win(scene.true_fg, size, c), scene.win(scene.regions, size, c)
                ts, tc = scene.win(scene.true_sat, size, c), scene.win(scene.true_comp, size, c)
                rows.append({
                    "experiment": e.name, "setting": label_name, "window_px": size,
                    "labelled_px": int(found.sum()), "false_px": int((found & ~tf).sum()),
                    "labelled_share": float(found.mean()),
                    "outline_iou": (found & tf).sum() / max((found | tf).sum(), 1) if tf.any() else np.nan,
                    "ari_materials": ari(tr[tf], lab[tf]) if tf.sum() > 1 else np.nan,
                    "satellite_iou": (found & ts).sum() / max((found | ts).sum(), 1) if ts.any() else np.nan,
                    "satellite_px_found": int((found & ts).sum()), "satellite_px_true": int(ts.sum()),
                    "companion_px_found": int((found & tc).sum()), "companion_px_true": int(tc.sum()),
                    "share_object_on_found": float(objectness[found].mean()) if found.any() else np.nan,
                    "share_material_on_found": float(agreement[found].mean()) if found.any() else np.nan})
                maps[(e.name, label_name)] = {"labels": lab, "objectness": objectness, "agreement": agreement,
                                              "true_fg": tf, "regions": tr}
    return pd.DataFrame(rows), maps


def fits_scores(experiments, size=None):
    """The table of fits_results (kept for callers of the first version)."""
    return fits_results(experiments)[0]


# ----------------------------------------------------------------------------- main
def run_synthetic(exps, args, out):
    print("synthetic scenes at different brightness")
    df = synthetic_scores(exps, args.n_scenes, args.seed)
    df.to_csv(out / "synthetic_by_run.csv", index=False)
    summary = (df.groupby(["scenes", "stage", "log10_gain", "experiment"])[["material_acc", "foreground_iou", "false_fg_share"]]
                 .mean().round(3).reset_index())
    summary.to_csv(out / "synthetic_summary.csv", index=False)
    plot_synthetic(df, out / "synthetic_vs_gain.png")
    print("robustness: band curve, empty scenes, halo, artifacts")
    rb = robustness_scores(exps, args.n_scenes, args.seed)
    rb.to_csv(out / "robustness_by_run.csv", index=False)
    (rb.groupby(["test", "level", "experiment"])[["material_acc", "foreground_iou", "false_fg_share", "labelled_share"]]
       .mean().round(4).reset_index().to_csv(out / "robustness_summary.csv", index=False))
    plot_robustness(rb, out / "robustness.png")
    return df, rb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", required=True, help="folders under --logdir; the first sets the scene config")
    ap.add_argument("--logdir", default=str(ROOT / "logs"))
    ap.add_argument("--max-runs", type=int, default=None, help="use only the first N runs of each experiment")
    ap.add_argument("--n-scenes", type=int, default=200, help="synthetic scenes per kind and level")
    ap.add_argument("--seed", type=int, default=777, help="first key of the scene seeds, apart from every training and test seed")
    ap.add_argument("--no-fits", action="store_true", help="skip the estimate.fits part")
    ap.add_argument("--out", default=str(ROOT / "outputs/brightness_eval"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    exps = [Experiment(args.logdir, name, args.max_runs) for name in args.experiments]
    for e in exps:
        print(f"{e.name}: {len(e.models)} runs, input_norm={e.norm}, "
              f"gain_log10_range={e.config.data.params.sim.get('gain_log10_range', None)}")

    df, _ = run_synthetic(exps, args, out)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.assign(experiment=df.experiment.map(short))[(df.scenes == "satellite")].groupby(
            ["stage", "log10_gain", "experiment"]).material_acc.mean().unstack().round(3))

    if not args.no_fits:
        print("estimate.fits against object.fits")
        ft, _ = fits_results(exps)
        ft.to_csv(out / "estimate_fits.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 400):
            print(ft.round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
