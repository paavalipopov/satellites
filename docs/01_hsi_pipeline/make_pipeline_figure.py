"""Schematic of the synthetic HSI generation pipeline.

Stages run top to bottom; each card lists what the stage does (left) and
what it generates (right).  Thumbnails come from a real sample and all
numbers from SimConfig, so the figure tracks the code.

    python scripts/make_pipeline_figure.py      # -> figures/hsi_pipeline.{pdf,svg,png}
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src import SimConfig, SpectralLibrary          # noqa: E402
from src.pipeline import generate_sample            # noqa: E402

SEED = 3                     # layout 2: panels, bus, boom, dish

# palette
INK, INK2, MUTED = "#1c1c1e", "#4a4a4f", "#8a8a8f"
CARD, EDGE, PANEL = "#f6f6f4", "#d6d6d1", "#ffffff"
ONCE_TAG, SAMPLE_TAG, OUT_FILL, OUT_EDGE = "#e7e7e3", "#dde9f8", "#eef3fb", "#b9cbe6"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BG_CELL = "#ececE8"
RAMP = ("#f3f5f8", "#1f4e8c")                        # one hue, light -> dark

# layout, in data units (1000 wide, y down)
W, H = 1000, 1108
X0, X1 = 92, 940                                     # cards
PX0, PX1 = 588, 926                                  # "generates" panels
CX = PX0 + 172                                       # panel caption column
TX = X0 + 48                                         # body text column
LANE_L, LANE_R, SPINE = 52, 972, 330
S0, S1, S2, S3, OUT = (18, 176), (246, 192), (490, 214), (756, 220), (1028, 62)

# TrueType in PDF (journals reject Type 3); live, editable text in SVG
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none"})

LIMITS = []                  # (text artist, rightmost allowed x)


# ----------------------------------------------------------------- text
def text(ax, x, y, s, size=8.2, color=INK, weight="normal", ha="left", va="center",
         limit=None, **kw):
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)
    if limit is not None:
        LIMITS.append((t, limit))
    return t


def width(ax, s, size, weight="normal"):
    """Rendered width of a string, in data units."""
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    bb = t.get_window_extent(ax.figure.canvas.get_renderer())
    t.remove()
    inv = ax.transData.inverted()
    return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]


def check_overflow(ax):
    r, inv = ax.figure.canvas.get_renderer(), ax.transData.inverted()
    bad = [(inv.transform((t.get_window_extent(r).x1, 0))[0] - lim, t.get_text())
           for t, lim in LIMITS]
    bad = [(d, s) for d, s in bad if d > 0]
    for d, s in bad:
        print(f"  overflow by {d:4.1f}: {s!r}")
    return not bad


# ----------------------------------------------------------------- blocks
def card(ax, st, num, title, scope, rnd):
    y, h = st
    ax.add_patch(FancyBboxPatch((X0, y), X1 - X0, h, boxstyle="round,pad=0,rounding_size=12",
                                fc=CARD, ec=EDGE, lw=0.9))

    # badge, title, scope tag
    ax.add_patch(Circle((X0 + 25, y + 24), 13, fc=INK, ec="none"))
    text(ax, X0 + 25, y + 24.5, num, size=8.5, color="white", weight="bold", ha="center")
    text(ax, TX, y + 24, title, size=10, weight="bold")
    tx = TX + width(ax, title, 10, "bold") + 12
    tw = width(ax, scope, 7) + 14
    ax.add_patch(FancyBboxPatch((tx, y + 15), tw, 18, boxstyle="round,pad=0,rounding_size=9",
                                fc=ONCE_TAG if scope == "once" else SAMPLE_TAG, ec="none"))
    text(ax, tx + tw / 2, y + 24.5, scope, size=7, color=INK2, ha="center")

    # where randomness enters
    text(ax, X1 - 16, y + 24, rnd, size=7, color=MUTED, ha="right", style="italic")


def panel(ax, st, caption):
    """White 'generates' box; caption rows are (text, style) or a legend line."""
    y, h = st[0] + 44, st[1] - 56
    ax.add_patch(FancyBboxPatch((PX0, y), PX1 - PX0, h, boxstyle="round,pad=0,rounding_size=8",
                                fc=PANEL, ec=EDGE, lw=0.8))
    text(ax, PX0 + 12, y + 14, "GENERATES", size=6.3, color=MUTED, weight="bold")
    for k, (s, kw) in enumerate(caption):
        yy = y + 40 + 17 * k
        if "line" in kw:                             # legend swatch
            ax.plot([CX, CX + 16], [yy, yy], color=MUTED, lw=1.3, ls=kw.pop("line"))
            text(ax, CX + 22, yy, s, size=7.2, color=INK2, limit=PX1 - 8, **kw)
        else:
            text(ax, CX, yy, s, limit=PX1 - 8, **kw)
    return y


def lines(ax, y, rows, dy=21):
    for k, row in enumerate(rows):
        s, kw = row if isinstance(row, tuple) else (row, {})
        text(ax, TX, y + dy * k, s, limit=PX0 - 14, **kw)


def substep(ax, y, tag, name, formula, notes):
    ax.add_patch(FancyBboxPatch((TX, y - 9), 26, 18, boxstyle="round,pad=0,rounding_size=4",
                                fc="#e2e2de", ec="none"))
    text(ax, TX + 13, y + 0.5, tag, size=7.3, color=INK2, weight="bold", ha="center")
    text(ax, TX + 36, y, name, size=8.4, weight="bold")
    text(ax, TX + 36 + width(ax, name, 8.4, "bold") + 14, y, formula, size=8.6, limit=PX0 - 14)
    for k, note in enumerate(notes):
        text(ax, TX + 36, y + 19 * (k + 1), note, size=7.4, color=INK2, limit=PX0 - 14)


def arrow(ax, x0, y0, x1, y1, label=None, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color, mutation_scale=9,
                                shrinkA=0, shrinkB=0))
    if label:
        text(ax, x0 + 10, (y0 + y1) / 2, label, size=7.6, color=INK2, style="italic")


def lane(ax, pts, label, lpos, rot):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=INK, lw=1.0, solid_capstyle="butt")
    arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1])
    text(ax, *lpos, label, size=7.6, color=INK2, style="italic", rotation=rot, ha="center")


# ----------------------------------------------------------------- thumbnails
def ramp(v, vmax):
    t = np.clip(np.asarray(v, float) / vmax, 0, 1)[..., None]
    a, b = np.array(to_rgb(RAMP[0])), np.array(to_rgb(RAMP[1]))
    return a + t * (b - a)


def grid(ax, x, y, cell, rgb):
    n = rgb.shape[0]
    cells = [Rectangle((x + j * cell, y + i * cell), cell, cell) for i in range(n) for j in range(n)]
    ax.add_collection(PatchCollection(cells, facecolor=rgb.reshape(-1, 3), edgecolor="white",
                                      linewidth=0.25))
    ax.add_patch(Rectangle((x, y), n * cell, n * cell, fc="none", ec=INK2, lw=0.6))


def mask_rgb(mask, color):
    rgb = np.tile(to_rgb(BG_CELL), (*mask.shape, 1))
    for mid, c in color.items():
        rgb[mask == mid] = to_rgb(c)
    return rgb


def stack(ax, x, y, cell, cube, bands, vmax, off=17):
    """Band slices back to front, offset up-right like a deck of cards."""
    n = len(bands)
    for k, b in enumerate(bands):
        d = (n - 1 - k) * off
        grid(ax, x + d, y - d, cell, ramp(cube[b], vmax))

    # wavelength axis along the lower-right diagonal
    s, span = cube.shape[1] * cell, (n - 1) * off
    ax.annotate("", xy=(x + s + 8 + span, y + s + 2 - span), xytext=(x + s + 8, y + s + 2),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=MUTED, mutation_scale=7))
    text(ax, x + s + 8 + span / 2 + 9, y + s + 2 - span / 2 + 6, "λ", size=8, color=MUTED)


def spectra(ax, x, y, w, h, curves, lam):
    ax.plot([x, x + w], [y + h, y + h], color=EDGE, lw=0.8)
    ax.plot([x, x], [y, y + h], color=EDGE, lw=0.8)
    for s, color, ls in curves:
        ax.plot(x + (lam - lam[0]) / (lam[-1] - lam[0]) * w, y + h - np.clip(s, 0, 1) * h,
                color=color, lw=1.3, ls=ls)
    text(ax, x, y + h + 9, f"{lam[0]:.0f}", size=6.5, color=MUTED)
    text(ax, x + w, y + h + 9, f"{lam[-1]:.0f} nm", size=6.5, color=MUTED, ha="right")
    text(ax, x - 5, y + h, "0", size=6.5, color=MUTED, ha="right")
    text(ax, x - 5, y, "1", size=6.5, color=MUTED, ha="right")


# ----------------------------------------------------------------- figure
def main():
    warnings.simplefilter("ignore")
    cfg = SimConfig()
    lib = SpectralLibrary(cfg, root=ROOT)
    s = generate_sample(cfg, SEED, lib, scene="satellite")

    lam, sig = cfg.wavelengths_nm, cfg.psf_sigma_px
    nb, sz = cfg.n_bands, cfg.image_size
    color = {mid: CAT[k] for k, mid in enumerate(sorted(s.material_ids))}
    n_irr = sum(m.irradiated is not None for m in lib.materials.values())
    vmax = max(s.cube_noisy_spectra.max(), 1e-6)
    bands = [0, nb // 2, nb - 1]
    yb = lambda st: st[0] + st[1]

    fig = plt.figure(figsize=(7.5, 7.5 * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # ---- stage 0: spectral library
    card(ax, S0, "0", "Spectral library", "once", "deterministic")
    lines(ax, S0[0] + 60, [
        "Read 46 measured DHR spectra from 3 sheets",
        (f"bus 30 · wings 8 · antenna 8;  {n_irr} with an irradiated trace",
         dict(color=INK2, size=7.6)),
        f"Resample onto the grid: {nb} bands, {lam[0]:.0f}–{lam[-1]:.0f} nm",
        (f"Δλ = {lam[1] - lam[0]:.2f} nm, linear interpolation", dict(color=INK2, size=7.6)),
        "Check every value in [0, 1]  →  warn · raise · clip",
    ])
    py = panel(ax, S0, [
        ("material library", dict(weight="bold", size=8.2)),
        ("46 materials", dict(size=7.6, color=INK2)),
        (f"{nb} bands each", dict(size=7.6, color=INK2)),
        ("pristine", dict(line="-")),
        ("irradiated", dict(line=(0, (3, 2)))),
    ])

    # clean examples only: skip spectra the validator flagged
    flagged = {ib.material for ib in lib.invalid}
    picks = [m for m in sorted(s.material_ids)
             if lib.materials[s.material_names[m]].irradiated is not None
             and s.material_names[m] not in flagged][:3]
    curves = []
    for mid in picks:
        m = lib.materials[s.material_names[mid]]
        curves += [(m.pristine, color[mid], "-"), (m.irradiated, color[mid], (0, (3, 2)))]
    spectra(ax, PX0 + 30, py + 30, 118, 58, curves, lam)

    # ---- stage 1: scene mask
    card(ax, S1, "1", "Scene mask", "per sample", "random: scene type · layout · materials · pose")
    lines(ax, S1[0] + 60, [
        "Pick a scene: satellite 30% · shapes 30% · patches 40%",
        "Satellite: layout 1 or 2 (50/50), one material per part",
        ("3 bus · 1 antenna · 1 wings, each from its own group", dict(color=INK2, size=7.6)),
        "Random pose: tilt 45–135°, spin ±180°",
        f"Pinhole camera, {cfg.fov_deg:.0f}° FOV  →  z-buffer rasterise",
        ("writes a material id per pixel, not a brightness", dict(color=INK2, size=7.6)),
    ], dy=20)
    py = panel(ax, S1, [
        ("mask M = label", dict(weight="bold", size=8.2)),
        (f"{sz} × {sz}, uint8", dict(size=7.6, color=INK2)),
        ("0 = background", dict(size=7.6, color=INK2)),
        ("1–46 = material id", dict(size=7.6, color=INK2)),
    ])
    grid(ax, PX0 + 30, py + 26, 6.4, mask_rgb(s.mask, color))

    # ---- stage 2: cube construction
    card(ax, S2, "2", "Cube construction", "per sample", "random: w · material noise")
    y = S2[0] + 62
    substep(ax, y, "2a", "Blend",
            r"$s_m = (1-w)\,s_m^{\rm pris} + w\,s_m^{\rm irr}$",
            ["one w ~ U(0, 1) per scene, shared by all materials"])
    substep(ax, y + 50, "2b", "Spectral noise",
            r"$\tilde s_m = s_m + \mathcal{N}\!\left(0,\,(\frac{1}{2}\,{\rm std}\,s_m)^2\right)$",
            ["one draw per material, shared by all its pixels"])
    substep(ax, y + 100, "2c", "Paint",
            r"$x[:,\,i,\,j] = \tilde s_{M(i,j)}$",
            ["copy down the λ axis; background stays 0"])
    panel(ax, S2, [
        ("cube x", dict(weight="bold", size=8.2)),
        (f"{nb} × {sz} × {sz}", dict(size=7.6, color=INK2)),
        ("same pattern in", dict(size=7.6, color=INK2)),
        ("every band", dict(size=7.6, color=INK2)),
        ("+ clean copy", dict(size=7.0, color=MUTED)),
        ("→ regression target", dict(size=7.0, color=MUTED)),
    ])
    stack(ax, PX0 + 26, S2[0] + 112, 5.4, s.cube_noisy_spectra, bands, vmax)

    # ---- stage 3: degradation
    card(ax, S3, "3", "Degradation", "per sample", "random: sensor noise")
    y = S3[0] + 62
    substep(ax, y, "3a", "PSF blur",
            r"$\sigma_j = (k\,\lambda_j/D)\,/\,(p/f)$",
            [f"per-band Gaussian, σ grows {sig[0]:.2f} → {sig[-1]:.2f} px",
             f"k = {cfg.psf_k:g},  D = {cfg.aperture_m:g} m,  p = {cfg.pixel_pitch_m * 1e6:g} µm,"
             f"  f = {cfg.focal_length_m:g} m"])
    substep(ax, y + 70, "3b", "Sensor noise",
            r"$x \leftarrow x + \mathcal{N}(0,\,\sigma_n^2),\ \ \sigma_n = $" + f"{cfg.sensor_noise_std:g}",
            ["independent per pixel and band"])
    text(ax, TX + 36, y + 128, "the blur is the only step coupling space and wavelength",
         size=7.2, color=MUTED, style="italic", limit=PX0 - 14)
    panel(ax, S3, [
        ("degraded cube", dict(weight="bold", size=8.2)),
        (f"{nb} × {sz} × {sz}", dict(size=7.6, color=INK2)),
        ("float, unnormalised", dict(size=7.6, color=INK2)),
        ("= model input", dict(size=7.6, color=INK2, weight="bold")),
    ])
    stack(ax, PX0 + 26, S3[0] + 112, 5.4, s.cube, bands, vmax)

    # ---- output: training pair
    ax.add_patch(FancyBboxPatch((X0, OUT[0]), X1 - X0, OUT[1],
                                boxstyle="round,pad=0,rounding_size=12", fc=OUT_FILL, ec=OUT_EDGE, lw=0.9))
    text(ax, X0 + 22, OUT[0] + 31, "Training pair", size=10, weight="bold")
    text(ax, 300, OUT[0] + 22, f"x  =  degraded cube   {nb} × {sz} × {sz}", size=8.2)
    text(ax, 300, OUT[0] + 41, "input", size=7.4, color=INK2)
    text(ax, 620, OUT[0] + 22, f"y  =  mask   {sz} × {sz}", size=8.2)
    text(ax, 620, OUT[0] + 41, "target, 47 classes incl. background", size=7.4, color=INK2,
         limit=X1 - 10)

    # ---- spine: what each stage hands to the next
    arrow(ax, SPINE, yb(S0), SPINE, S1[0], "material ids, by part group")
    arrow(ax, SPINE, yb(S1), SPINE, S2[0], "mask M")
    arrow(ax, SPINE, yb(S2), SPINE, S3[0], "cube x")
    arrow(ax, SPINE, yb(S3), SPINE, OUT[0], "degraded x")

    # ---- side lanes: spectra skip stage 1, the mask skips stages 2-3
    ys, ye = S0[0] + 96, S2[0] + 62
    lane(ax, [(X0, ys), (LANE_L, ys), (LANE_L, ye), (X0, ye)],
         "spectra", (LANE_L - 11, (ys + ye) / 2), 90)
    ym, yo = S1[0] + 110, OUT[0] + 31
    lane(ax, [(X1, ym), (LANE_R, ym), (LANE_R, yo), (X1, yo)],
         "mask  =  label", (LANE_R + 12, (ym + yo) / 2), 270)

    ok = check_overflow(ax)
    out = os.path.join(ROOT, "figures", "hsi_pipeline")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300 if ext == "png" else None, facecolor="white")
    print("wrote", out + ".{pdf,svg,png}", "| layout ok" if ok else "| LAYOUT OVERFLOW",
          "| spectra shown:", [s.material_names[m] for m in picks])


if __name__ == "__main__":
    main()
