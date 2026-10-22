"""
Measured spectral library.

Three spreadsheets hold directed-hemispheric reflectance (DHR) for 46 spacecraft
materials, each with a pristine and (usually) an electron-irradiated trace.
This module loads them, resamples onto the config's wavelength grid, and exposes
the pristine/irradiated pair per material.

The blend between the two states happens later, per scene, in `pipeline`.
"""

from __future__ import annotations

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
_MARGINAL = 0.05     # warning text only: outside [-0.05, 1.05] counts as "severe"
_EDGE_BANDS = 3      # warning text only: within 3 bands of a grid end counts as "edge"
_GAP_NM = 10.0       # unmeasured bands closer than this to a measurement are not reported # TODO 'not reported' sounds vague. does it raise warnings? what can silently happen with this gap?


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


@dataclass
class InvalidBand: # TODO: this class and UnmeasuredBand below make me confused, I don't understand their role in the script. if they are mostly used for reporting, can you add notes to it, or just explain in a docstring what they do? be brief, this file is already long enough. can we trim them without looksing functionality/clarity?
    """One band of one spectrum that falls outside physical reflectance."""
    material: str
    state: str                      # "pristine" | "irradiated"
    band: int
    wavelength_nm: float
    value: float


@dataclass
class UnmeasuredBand:
    """One grid band outside a spectrum's measured wavelength range."""
    material: str
    state: str                      # "pristine" | "irradiated"
    band: int
    wavelength_nm: float
    side: str                       # "blue" (below the scan) | "red" (above it)
    edge_nm: float                  # nearest measured wavelength


class SpectralLibrary:
    """Materials resampled onto one wavelength grid."""

    def __init__(self, cfg: SimConfig, root: str = "."):
        self.cfg = cfg
        self.grid = cfg.wavelengths_nm
        self.materials: dict[str, Material] = {}
        self.invalid: list[InvalidBand] = []
        self.unmeasured: list[UnmeasuredBand] = []
        self._load(os.path.join(root, cfg.material_dir))
        self._report()

    # ------------------------------------------------------------------
    def _load(self, path: str) -> None:
        for group, fname in _SHEET_FILES.items():
            sheets = pd.read_excel(os.path.join(path, fname), sheet_name=None)
            for name, df in sheets.items():
                if name not in NAME_TO_ID:
                    continue                       # ignore sheets not in the taxonomy
                pris, irr, ranges = self._resample(df)
                if pris is None:
                    continue
                for state, v in (("pristine", pris), ("irradiated", irr)):
                    if v is None:
                        continue
                    bad = np.where((v < -_TOL) | (v > 1.0 + _TOL))[0]
                    self.invalid += [
                        InvalidBand(name, state, int(b), float(self.grid[b]), float(v[b]))
                        for b in bad
                    ]
                    self.unmeasured += self._unmeasured(name, state, *ranges[state])
                self.materials[name] = Material(
                    name=name, id=NAME_TO_ID[name], group=group,
                    pristine=self._apply_policy(pris),
                    irradiated=self._apply_policy(irr),
                    pristine_raw=pris, irradiated_raw=irr,
                )
        missing = set(NAME_TO_ID) - set(self.materials)
        if missing:
            raise RuntimeError(f"materials missing from the spreadsheets: {sorted(missing)}")

    def _resample(self, df: pd.DataFrame):
        """Columns are (wavelength_nm, pristine, irradiated).  Columns 3+ hold
        no numeric data — they are blank spacers plus a text note giving the
        irradiation dose."""
        d = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
        d = d.dropna(subset=[d.columns[0], d.columns[1]])
        if d.empty:
            return None, None, None
        lam = d.iloc[:, 0].to_numpy(float)
        order = np.argsort(lam)
        lam = lam[order]

        def interp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """Resample only — the invalid-value policy is applied later, so
            that validation sees what the spreadsheet actually says."""
            right = 0.0 if self.cfg.extrapolate_red == "zero" else y[-1]
            return np.interp(self.grid, x, y, left=y[0], right=right)

        pris = interp(lam, d.iloc[:, 1].to_numpy(float)[order])
        ranges = {"pristine": (lam[0], lam[-1])}
        irr = None
        if d.shape[1] >= 3:
            raw = d.iloc[:, 2].to_numpy(float)[order]
            ok = np.isfinite(raw)
            if ok.sum() >= 2:
                irr = interp(lam[ok], raw[ok])
                ranges["irradiated"] = (lam[ok][0], lam[ok][-1])
        return pris, irr, ranges

    def _unmeasured(self, name: str, state: str, lo: float, hi: float) -> list[UnmeasuredBand]: # TODO what's the role of UnmeasuredBand in the SpectralLib? are they shown during report?
        """Grid bands more than _GAP_NM outside the measured range [lo, hi]."""
        out = []
        for b, lam in enumerate(self.grid):
            if lam < lo - _GAP_NM:
                out.append(UnmeasuredBand(name, state, b, float(lam), "blue", float(lo)))
            elif lam > hi + _GAP_NM:
                out.append(UnmeasuredBand(name, state, b, float(lam), "red", float(hi)))
        return out

    # ------------------------------------------------------------------
    def _apply_policy(self, v):
        if v is None:
            return None
        return np.clip(v, 0.0, 1.0) if self.cfg.clip_reflectance else v

    def _report(self) -> None:
        """Always detect; `on_invalid` and `on_unmeasured` decide whether to speak up."""
        checks = ((self.invalid, self.cfg.on_invalid, self.invalid_summary),
                  (self.unmeasured, self.cfg.on_unmeasured, self.unmeasured_summary))
        for found, policy, summary in checks:
            if not found or policy == "ignore":
                continue
            if policy == "raise":
                raise ValueError(summary())
            warnings.warn(summary(), SpectralLibraryWarning, stacklevel=3)

    def invalid_summary(self) -> str:
        """Human-readable account of every non-physical value on this grid."""
        if not self.invalid:
            return "all spectra lie within [0, 1] on this grid"
        by_mat: dict[tuple[str, str], list[InvalidBand]] = {}
        for ib in self.invalid:
            by_mat.setdefault((ib.material, ib.state), []).append(ib)
        bands = sorted({ib.band for ib in self.invalid})
        lines = [
            f"{len(self.invalid)} non-physical reflectance values across "
            f"{len(by_mat)} spectra, on the {self.cfg.n_bands}-band "
            f"{self.cfg.lambda_min_nm:.0f}-{self.cfg.lambda_max_nm:.0f} nm grid.",
            f"  affected bands: {bands}  "
            f"(lambda {[round(float(self.grid[b]), 1) for b in bands]} nm)",
        ]
        n_edge = sum(1 for ib in self.invalid
                     if ib.band < _EDGE_BANDS or ib.band >= self.cfg.n_bands - _EDGE_BANDS)
        n_severe = sum(1 for ib in self.invalid
                       if ib.value < -_MARGINAL or ib.value > 1.0 + _MARGINAL)
        lines.append(f"  {n_severe} severe (outside [{-_MARGINAL:.2f}, {1+_MARGINAL:.2f}]), "
                     f"{len(self.invalid) - n_severe} marginal; "
                     f"{n_edge} at a grid edge, {len(self.invalid) - n_edge} in the interior.")
        worst = sorted(by_mat.items(),
                       key=lambda kv: -max(abs(x.value) for x in kv[1]))
        for (mat, state), items in worst[:10]:
            ex = max(items, key=lambda x: abs(x.value))
            lines.append(f"  {mat:20s} {state:11s} {len(items)} band(s), worst "
                         f"{ex.value:+8.2f} at band {ex.band} "
                         f"({ex.wavelength_nm:.1f} nm)")
        if len(worst) > 10:
            lines.append(f"  ... and {len(worst) - 10} more spectra")
        if n_severe:
            lines.append(
                "  Severe values at a scan edge are instrument artifacts: DHR is a "
                "ratio, and near the UV cutoff both signal and reference approach "
                "zero, so the quotient is noise. These carry no information — "
                "trim the grid past them rather than clipping.")
        if len(self.invalid) - n_severe:
            lines.append(
                "  Marginal values (just outside [0, 1], often mid-band) are usually "
                "a calibration offset against the reference standard, not junk — "
                "clipping them is reasonable.")
        if self.cfg.clip_reflectance:
            lines.append(
                "  clip_reflectance=True: all of the above were clipped to [0, 1], "
                "which substitutes a fabricated value wherever the measurement was "
                "actually missing.")
        else:
            lines.append("  clip_reflectance=False: these enter the cubes unaltered.")
        return "\n".join(lines)

    def unmeasured_summary(self) -> str:
        """Human-readable account of grid bands that no measurement covers."""
        if not self.unmeasured:
            return "every grid band is covered by its spectrum's measurements"
        by_trace: dict[tuple[str, str, str], list[UnmeasuredBand]] = {}
        for ub in self.unmeasured:
            by_trace.setdefault((ub.material, ub.state, ub.side), []).append(ub)
        lines = [
            f"{len(self.unmeasured)} grid bands across {len(by_trace)} spectra lie more than "
            f"{_GAP_NM:.0f} nm outside the measured range, on the {self.cfg.n_bands}-band "
            f"{self.cfg.lambda_min_nm:.0f}-{self.cfg.lambda_max_nm:.0f} nm grid.",
        ]
        red_fill = "set to 0" if self.cfg.extrapolate_red == "zero" else "held at the last value"
        for (mat, state, side), items in sorted(by_trace.items()):
            edge = items[0].edge_nm
            far = max(abs(ub.wavelength_nm - edge) for ub in items)
            where, fill = ("below", "held at the first value") if side == "blue" else ("above", red_fill)
            lines.append(f"  {mat:20s} {state:11s} {len(items)} band(s) {where} {edge:.0f} nm "
                         f"(up to {far:.0f} nm away), {fill}")
        lines.append("  These values are extrapolated, not measured: trim the grid or drop "
                     "the material if they matter.")
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
