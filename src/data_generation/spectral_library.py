"""
Measured spectral library.

Three spreadsheets hold directed-hemispheric reflectance (DHR) for 46 spacecraft
materials, each with a pristine and (usually) an electron-irradiated trace.
This module loads them, resamples onto the config's wavelength grid, and exposes
the pristine/irradiated pair per material.

The blend between the two states happens later, per scene, in `pipeline`.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimConfig

# ----------------------------------------------------------------------
# Material index <-> name.  Index 0 is background; 1..46 are materials.
# Grouped by which part of the spacecraft they may be assigned to.
# TODO for later: think about how to organize the materials without variables, 
# by just organizing the material data into something more readable and getting the 
# names from there. Something like csv/json files? It should be compatible with the 
# config params like require_irradiated that does not mix the material indices.

BUS = [
    "LCP", "Thermalbright_N", "Melinex", "Mylar", "CORIN_XLS", "Kapton-MT",
    "Kapton-HN", "Solar_cell", "Black paint", "Kapton_CRC", "Kapton_CS)",
    "Kapton-WS-glossy", "Kapton-FMT", "Kapton-PV9101", "Kapton-PV9103",
    "Mylar_aluminized", "Aluminum_Foil", "Kapton HN 1 mil", "AZW-LAII",
    "AZ-2000", "AZ-400", "AZJ-4020", "AZ-3700", "AZ-2100", "AZ-93",
    "Stainless_steel_1", "Stainless_Steel_2", "BetaCloth",
    "Aluminum 7075T6", "Aluminum 6061",
]
WINGS = [
    "Fused_silica", "CMX", "0214", "Glass_Fiber", "CMG_with_Al",
    "CMG_with_Cu", "Si solar cell", "Ge solar cell",
]
ANTENNA = [
    "GFRP", "CFRP", "Elgiloy", "Aluminum", "Titanium",
    "Copper_foil", "Copper", "Gold_foil",
]

MATERIAL_NAMES = BUS + WINGS + ANTENNA                 # index i -> id i+1
NAME_TO_ID = {n: i + 1 for i, n in enumerate(MATERIAL_NAMES)}
ID_TO_NAME = {i + 1: n for i, n in enumerate(MATERIAL_NAMES)}
GROUPS = {"bus": BUS, "wings": WINGS, "antenna": ANTENNA}
N_CLASSES = len(MATERIAL_NAMES) + 1                    # 47, including background

_SHEET_FILES = {"bus": "Bus.xlsx", "wings": "Wings.xlsx", "antenna": "antenna.xlsx"}

_TOL = 1e-6          # ignore float dust around the [0, 1] boundary
_MARGINAL = 0.05     # invalid values beyond [-0.05, 1.05] count as "severe"
_GAP_NM = 10.0       # held values this close to a scan's end are not flagged; zero fills always are


class SpectralLibraryWarning(UserWarning):
    """Warns about library values that are not physical reflectances, or about
    grid bands that no measurement covers."""

# Duplicate / near-duplicate pairs measured at 19-49% mutual confusion.
# Kept as a note rather than acted on: several of the 47 "classes" are the
# same physical material, which makes the label set partly ill-posed.
KNOWN_DUPLICATES = [
    ("GFRP", "Glass_Fiber"), ("Aluminum", "Aluminum 6061"),
    ("Gold_foil", "Ge solar cell"), ("CFRP", "Black paint"),
    ("0214", "Fused_silica"), ("Kapton-FMT", "Kapton-MT"),
    ("Kapton_CRC", "Kapton-HN"),
]

# Two sheets carry values that are suspicious for the material named.
# Gold reflects ~0.97 in the NIR; `Gold_foil` reads 0.002-0.006 across the whole
# range with no gold-like rise at 500-600 nm.  Copper reflects ~0.97 above
# 700 nm; `Copper_foil` peaks at 0.25.  The separate `Copper` sheet
# (0.26/0.65/0.74/0.81/0.84 at 450/600/800/1500/2400 nm) has the right shape and
# magnitude, so the instrument was capable of measuring it.
SUSPECT_SHEETS = {
    "Gold_foil":   "reads 0.002-0.006; bulk gold is ~0.97 in the NIR",
    "Copper_foil": "peaks at 0.25; bulk copper is ~0.97 above 700 nm",
}

@dataclass
class Material:
    name: str
    id: int
    group: str
    pristine: np.ndarray            # [n_bands], after the invalid-value policy
    irradiated: np.ndarray | None   # [n_bands] or None
    pristine_raw: np.ndarray = None       # before clipping — what the sheet says
    irradiated_raw: np.ndarray = None
    measured: dict = None           # state -> (wavelength_nm, values) as read from the sheet


@dataclass
class FlaggedBand:
    """One grid band of one spectrum flagged while loading.  Collected in
    `lib.flags`; the warnings and `plot_materials` are built from these."""
    material: str
    state: str                      # "pristine" | "irradiated"
    band: int
    wavelength_nm: float
    issue: str                      # "invalid": outside [0, 1] | "unmeasured": filled past the scan
    value: float                    # invalid: the value; unmeasured: nm past the scan (- blue, + red)

    @property
    def severe(self) -> bool:
        return self.issue == "invalid" and not -_MARGINAL <= self.value <= 1.0 + _MARGINAL


class SpectralLibrary:
    """Materials resampled onto one wavelength grid."""

    def __init__(self, cfg: SimConfig, root: str = "."):
        self.cfg = cfg
        self.grid = cfg.wavelengths_nm
        self.materials: dict[str, Material] = {}
        self.flags: list[FlaggedBand] = []
        self._load(os.path.join(root, cfg.material_dir))
        self._report()

    # ------------------------------------------------------------------
    def _load(self, path: str) -> None:
        for group, fname in _SHEET_FILES.items():
            sheets = pd.read_excel(os.path.join(path, fname), sheet_name=None)
            for name, df in sheets.items():
                if name not in NAME_TO_ID:
                    continue                       # ignore sheets not in the taxonomy
                measured = self._read(df)
                if measured is None:
                    continue

                raw = {state: self._resample(lam, y) for state, (lam, y) in measured.items()}
                for state, v in raw.items():
                    self.flags += self._flag(name, state, v, measured[state][0])

                self.materials[name] = Material(
                    name=name, id=NAME_TO_ID[name], group=group,
                    pristine=self._apply_policy(raw["pristine"]),
                    irradiated=self._apply_policy(raw.get("irradiated")),
                    pristine_raw=raw["pristine"], irradiated_raw=raw.get("irradiated"),
                    measured=measured,
                )
        missing = set(NAME_TO_ID) - set(self.materials)
        if missing:
            raise RuntimeError(f"materials missing from the spreadsheets: {sorted(missing)}")

    @staticmethod
    def _read(df: pd.DataFrame) -> dict | None:
        """state -> (wavelength_nm, values), sorted by wavelength.  Columns are
        (wavelength_nm, pristine, irradiated); columns 3+ hold no numeric data,
        only blank spacers and a text note giving the irradiation dose."""
        d = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
        d = d.dropna(subset=[d.columns[0], d.columns[1]])
        if d.empty:
            return None

        lam = d.iloc[:, 0].to_numpy(float)
        order = np.argsort(lam)
        lam = lam[order]
        out = {"pristine": (lam, d.iloc[:, 1].to_numpy(float)[order])}
        if d.shape[1] >= 3:
            irr = d.iloc[:, 2].to_numpy(float)[order]
            ok = np.isfinite(irr)
            if ok.sum() >= 2:
                out["irradiated"] = (lam[ok], irr[ok])
        return out

    def _resample(self, lam: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
        """Interpolate onto the grid: the blue end holds the first value, the red end
        follows `extrapolate_red`.  No clipping here, so validation sees the sheet."""
        right = 0.0 if self.cfg.extrapolate_red == "zero" else y[-1]
        return np.interp(self.grid if grid is None else grid, lam, y, left=y[0], right=right)

    def _flag(self, name: str, state: str, v: np.ndarray, lam: np.ndarray) -> list[FlaggedBand]:
        """Bands outside [0, 1], and bands filled past the scan (see _GAP_NM)."""
        out = [FlaggedBand(name, state, int(b), float(self.grid[b]), "invalid", float(v[b]))
               for b in np.where((v < -_TOL) | (v > 1.0 + _TOL))[0]]

        red_tol = 0.0 if self.cfg.extrapolate_red == "zero" else _GAP_NM
        for b, g in enumerate(self.grid):
            past = g - lam[0] if g < lam[0] else g - lam[-1] if g > lam[-1] else 0.0
            if past < -_GAP_NM or past > red_tol:
                out.append(FlaggedBand(name, state, b, float(g), "unmeasured", float(past)))
        return out

    def _apply_policy(self, v):
        if v is None:
            return None
        return np.clip(v, 0.0, 1.0) if self.cfg.clip_reflectance else v

    # ------------------------------------------------------------------ reporting
    def _report(self) -> None:
        """Always detect; `on_invalid` and `on_unmeasured` decide whether to speak up."""
        checks = (("invalid", self.cfg.on_invalid, self.invalid_summary),
                  ("unmeasured", self.cfg.on_unmeasured, self.unmeasured_summary))
        for issue, policy, summary in checks:
            if policy == "ignore" or not any(f.issue == issue for f in self.flags):
                continue
            if policy == "raise":
                raise ValueError(summary())
            warnings.warn(summary(), SpectralLibraryWarning, stacklevel=3)

    @property
    def _grid_desc(self) -> str:
        return f"{self.cfg.n_bands}-band {self.grid[0]:.0f}-{self.grid[-1]:.0f} nm grid"

    def invalid_summary(self) -> str:
        """Values outside [0, 1], worst spectra first, and what to do about them."""
        inv = [f for f in self.flags if f.issue == "invalid"]
        if not inv:
            return "all spectra lie within [0, 1] on this grid"
        by_trace: dict[tuple[str, str], list[FlaggedBand]] = {}
        for f in inv:
            by_trace.setdefault((f.material, f.state), []).append(f)
        n_sev = sum(f.severe for f in inv)

        lines = [f"{len(inv)} values outside [0, 1] in {len(by_trace)} spectra on the {self._grid_desc}: "
                 f"{n_sev} severe (beyond [{-_MARGINAL:.2f}, {1 + _MARGINAL:.2f}]), "
                 f"{len(inv) - n_sev} marginal."]
        worst = sorted(by_trace.items(), key=lambda kv: -max(abs(f.value) for f in kv[1]))
        for (mat, state), items in worst[:10]:
            ex = max(items, key=lambda f: abs(f.value))
            lines.append(f"  {mat:20s} {state:11s} {len(items)} band(s), worst {ex.value:+7.2f} "
                         f"at {ex.wavelength_nm:.0f} nm")
        if len(worst) > 10:
            lines.append(f"  ... and {len(worst) - 10} more spectra")

        lines += ["", "Severe values are scan-edge artifacts (DHR is a ratio of two near-zero "
                      "signals there); marginal ones are calibration offsets."]
        if n_sev:
            lines.append(f"To drop the severe ones, trim the grid: {self._clean_range()}.")
        lines.append("clip_reflectance=True clips all of them to [0, 1]: fine for marginal values, "
                     "a made-up value for severe ones." if self.cfg.clip_reflectance
                     else "clip_reflectance=False: they enter the cubes unaltered.")
        lines.append('lib.plot_materials("invalid") shows them.')
        return "\n".join(lines)

    def _clean_range(self) -> str:
        """Smallest trim of the grid ends (5 nm steps, same n_bands) that leaves no severe value."""
        n, lo, hi = self.cfg.n_bands, self.cfg.lambda_min_nm, self.cfg.lambda_max_nm
        cap = 0.25 * (hi - lo)                          # trim at most a quarter per end

        def severe(points):
            return any(((v < -_MARGINAL) | (v > 1.0 + _MARGINAL)).any()
                       for m in self.materials.values() for lam, y in m.measured.values()
                       for v in [self._resample(lam, y, points)])

        new_lo, new_hi = lo, hi
        while new_lo - lo < cap and severe(np.linspace(new_lo, hi, n)[: n // 2]):
            new_lo += 5.0
        while hi - new_hi < cap and severe(np.linspace(new_lo, new_hi, n)[n // 2:]):
            new_hi -= 5.0
        if severe(np.linspace(new_lo, new_hi, n)):
            return "not possible within a quarter of the range at either end"
        trims = [f"lambda_min_nm={new_lo:g}"] * (new_lo != lo) + [f"lambda_max_nm={new_hi:g}"] * (new_hi != hi)
        return ", ".join(trims)

    def unmeasured_summary(self) -> str:
        """Grid bands filled past a scan, per spectrum, and how to avoid them."""
        un = [f for f in self.flags if f.issue == "unmeasured"]
        if not un:
            return "every grid band is covered by its spectrum's measurements"
        by_trace: dict[tuple[str, str, bool], list[FlaggedBand]] = {}
        for f in un:
            by_trace.setdefault((f.material, f.state, f.value < 0), []).append(f)

        red_fill = "set to 0" if self.cfg.extrapolate_red == "zero" else "held at the last value"
        lines = [f"{len(un)} grid bands in {len(by_trace)} spectra lie past the measured scan "
                 f"on the {self._grid_desc}:"]
        starts, ends = [], []
        for (mat, state, blue), items in sorted(by_trace.items()):
            edge = items[0].wavelength_nm - items[0].value          # first or last measured nm
            far = max(abs(f.value) for f in items)
            (starts if blue else ends).append(edge)
            lines.append(f"  {mat:20s} {state:11s} {len(items)} band(s) {'below' if blue else 'above'} "
                         f"{edge:.0f} nm (up to {far:.0f} nm away), "
                         f"{'held at the first value' if blue else red_fill}")

        red_tol = 0.0 if self.cfg.extrapolate_red == "zero" else _GAP_NM
        fix = ([f"lambda_min_nm >= {math.ceil(max(starts) - _GAP_NM)}"] if starts else []) + \
              ([f"lambda_max_nm <= {math.floor(min(ends) + red_tol)}"] if ends else [])
        lines += ["", f"These values are extrapolated, not measured; to avoid them use {' and '.join(fix)}.",
                  'lib.plot_materials("unmeasured") shows them.']
        return "\n".join(lines)

    def blend(self, name: str, weight: float) -> np.ndarray:
        """Stage 0 output: one spectrum, somewhere between pristine and irradiated.

        s = (1 - w) * pristine + w * irradiated,   w in [0, 1]

        Materials with no irradiated trace ignore `weight` and stay pristine.
        """
        m = self.materials[name]
        if m.irradiated is None:
            return m.pristine.copy()
        return (1.0 - weight) * m.pristine + weight * m.irradiated

    # ------------------------------------------------------------------
    def usable(self, name: str) -> bool:
        """Does this material satisfy the current selection policy?"""
        if self.cfg.require_irradiated and self.materials[name].irradiated is None:
            return False
        return True

    @property
    def groups(self) -> dict[str, list[str]]:
        """Group -> selectable material names, after the selection policy.

        The original drew uniformly from every sheet that had a wavelength and
        a pristine column, with no quality or completeness filter of any kind.
        """
        return {g: [n for n in names if self.usable(n)] for g, names in GROUPS.items()}

    @property
    def usable_ids(self) -> np.ndarray:
        """Material ids the shape and patch scenes may draw from.  The original
        used `randint(1, 47)`, ignoring groups and availability alike."""
        return np.array(sorted(NAME_TO_ID[n] for n in self.materials if self.usable(n)))

    def ids_of(self, group: str) -> list[int]:
        return [NAME_TO_ID[n] for n in self.groups[group]]

    # ------------------------------------------------------------------ plotting
    def plot_materials(self, which="flagged"):
        """Measured scan (line) and grid values (dots) for each selected material,
        flagged bands circled; one panel per material, about sqrt(n) per row.

        `which`: a list of names, a group ("bus", "wings", "antenna"), "all", or a
        flag kind ("flagged", "invalid", "severe", "marginal", "unmeasured").
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        names = self._select(which)
        if not names:
            raise ValueError(f"no materials match {which!r}")
        ncol = math.ceil(math.sqrt(len(names)))
        nrow = math.ceil(len(names) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow + 0.4), squeeze=False)
        colour = {"pristine": "#2a78d6", "irradiated": "#eb6834", "invalid": "#d62728",
                  "unmeasured": "#6b6b70"}
        lo, hi, ylo, yhi = self.grid[0], self.grid[-1], -0.1, 1.2   # values beyond ylo/yhi sit on the edge

        for ax, name in zip(axes.flat, names):
            m = self.materials[name]
            for state, (lam, y) in m.measured.items():
                near = (lam > lo - 30) & (lam < hi + 30)
                ax.plot(lam[near], np.clip(y[near], ylo, yhi), color=colour[state], lw=0.8, alpha=0.5)
                ax.plot(self.grid, getattr(m, state), "o", color=colour[state], ms=2)

            mine = [f for f in self.flags if f.material == name]
            for f in mine:
                y = f.value if f.issue == "invalid" else getattr(m, f.state)[f.band]
                marker = "s" if f.issue == "unmeasured" else "^" if y > yhi else "v" if y < ylo else "o"
                ax.plot(f.wavelength_nm, np.clip(y, ylo, yhi), marker, mfc="none",
                        mec=colour[f.issue], ms=6, mew=1.1)

            counts = {"severe": sum(f.severe for f in mine),
                      "marginal": sum(f.issue == "invalid" and not f.severe for f in mine),
                      "unmeasured": sum(f.issue == "unmeasured" for f in mine)}
            label = " · ".join(f"{k} {v}" for k, v in counts.items() if v)
            worst = max((f.value for f in mine if f.issue == "invalid"), key=abs, default=0.0)
            if not ylo <= worst <= yhi:
                label += f" · worst {worst:+.1f}"
            ax.set_title(f"{name}\n{label}", fontsize=8, loc="left")
            ax.axhspan(0, 1, color="0.95", zorder=0)
            ax.set(xlim=(lo - 20, hi + 20), ylim=(ylo, yhi))
            ax.tick_params(labelsize=7)

        for ax in axes.flat[len(names):]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=colour[s], marker="o", ms=3, lw=0.8, label=s)
                   for s in ("pristine", "irradiated")]
        handles += [Line2D([], [], ls="", marker=mk, mfc="none", mec=colour[k], ms=6, label=lab)
                    for k, mk, lab in (("invalid", "o", "invalid"), ("invalid", "^", "invalid, off scale"),
                                       ("unmeasured", "s", "unmeasured"))]
        fig.legend(handles=handles, loc="upper right", ncol=5, fontsize=7, frameon=False)
        fig.supxlabel("wavelength (nm)", fontsize=8)
        fig.supylabel("reflectance", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
        return fig

    def _select(self, which) -> list[str]:
        """Material names for `plot_materials`: library order, or a list's own order."""
        if not isinstance(which, str):
            names = list(which)
            unknown = [n for n in names if n not in self.materials]
            if unknown:
                raise ValueError(f"unknown materials: {unknown}")
            return names
        if which == "all":
            return list(MATERIAL_NAMES)
        if which in GROUPS:
            return list(GROUPS[which])
        if which in self.materials:
            return [which]

        kinds = {"flagged": lambda f: True,
                 "invalid": lambda f: f.issue == "invalid",
                 "severe": lambda f: f.severe,
                 "marginal": lambda f: f.issue == "invalid" and not f.severe,
                 "unmeasured": lambda f: f.issue == "unmeasured"}
        if which not in kinds:
            raise ValueError(f"unknown selection {which!r}: use names, a group, 'all' or one of {list(kinds)}")
        hit = {f.material for f in self.flags if kinds[which](f)}
        return [n for n in MATERIAL_NAMES if n in hit]
