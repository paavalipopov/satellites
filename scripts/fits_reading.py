"""How the numbers in estimate.fits / object.fits are turned into what the model sees (a "reading").

The cubes hold DETECTED ELECTRONS per pixel, per wavelength slice, per 5 ms frame: the material's
reflectance multiplied by what the telescope and camera do to each colour. A "reading" divides that
back out, one number per slice. Three readings, chosen with a switch:

  solar           the OLD reading (notebook 06): divide by sunlight x band width,  E(lam) * dlam.
                  This stays the default, so every earlier result can be reproduced.
  radiometry      the collaborators' recipe (radiometry.py + recover_reflectance.py): divide by what the
                  simulator applied, G(lam) * lam * dlam   (atmosphere x optics x camera sensitivity,
                  photon counting, slice width). The sun is NOT in the truth cube, so it is not divided out.
  radiometry_sun  the same, also dividing by a 6000 K sun. This is the default of recover_reflectance.py
                  (TRUTH_INCLUDES_SUN = True). On this data it gives the wrong materials (see BRIGHTNESS.md);
                  it is here so that the switch can be tested, not because it is right.

Checked on this data: with `radiometry`, the satellite's 45 pixels reproduce the five material curves
of the reflectance figure and the layout of its diagram; `radiometry_sun` and `solar` do not.

The radiometry readings are RELATIVE reflectance: one overall constant is undetermined. `mag` only sets
that constant (it is the magnitude given to radiometry.calibrate(); the example's satellite is mag +6; a pixel's
reading is then about its reflectance times its share of the object's light, so values are of order 0.1).
Shape-only checks do not depend on it; a model that is given no scale does, which is why the by-hand
rows of the comparison are the clean test of the colour change.

scripts/radiometry.py is the file from the collaborators, unchanged (not written here: its author is
not named in the file). Its curves for the atmosphere, optics and camera are placeholders; the
recovery works on the simulated cubes because the simulation used the same placeholders.

As a script it does what recover_reflectance.py does, without touching assets/:

    python scripts/fits_reading.py                       # estimate.fits and object.fits -> outputs/fits_reading/
    python scripts/fits_reading.py --reading radiometry_sun --mag 1     # the collaborators' own defaults
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))      # so that `import radiometry` finds the sibling file

READINGS = ("solar", "radiometry", "radiometry_sun")

# the setup that made estimate.fits (recover_reflectance.py): must match the run that produced the file
D_CM = 360.0                    # telescope diameter [cm]
LAMB1, LAMB2, INCR = 400.0, 920.0, 8
T_SUN = 6000.0                  # K
ZENITH_DEG = 0.0
T_EXP_S, OBSTRUCTION = 5.0e-3, 0.0
DEFAULT_MAG = 6.0               # the example satellite's magnitude (new_input/message.txt); sets only the constant


def wavelength_grid(d_cm=D_CM, lamb1=LAMB1, lamb2=LAMB2, incr=INCR):
    """The wavelengths of the slices, in the order of the FITS files (red first), rebuilt as the main script does."""
    npts = int(2.0 * d_cm / 2)
    dim = int(2 ** int(np.log10(float(npts)) / np.log10(2.0) + 1))
    rad_max = int(dim / 4.0)
    rad_min = int(rad_max * (lamb1 / lamb2)) + 1
    nslice = int((rad_max - rad_min) / incr) + 1
    rad_min = rad_max - nslice * incr + incr
    rad = rad_min + incr * np.arange(nslice)
    return lamb1 * (dim / 4.0) / rad


def solar_divisor(lam_nm):
    """E * dlambda, the notebook-06 reading (sunlight above the atmosphere, ASTM G173, from pvlib)."""
    import pvlib
    lam = np.asarray(lam_nm, float)
    solar = pvlib.spectrum.get_reference_spectra(standard="ASTM G173-03")["extraterrestrial"]
    return np.interp(lam, solar.index, solar.values) * lam ** 2 / 6400


def radiometry_divisor(lam_nm, sun=False, mag=DEFAULT_MAG):
    """What the simulator did to each slice, per unit reflectance: G * lam * dlam (times the sun if sun=True).
    Taken from radiometry.calibrate() exactly as recover_reflectance.py does (dlam from np.gradient).

    Without the sun, the sun's units are divided out of the shape, which leaves the overall constant free. It is
    fixed here so that the numbers are the size a model can use: the divisor sums, over the slices, to the electrons
    of a gray object of reflectance 1 and magnitude `mag` (what calibrate() returns). The reading of a pixel is then
    about its reflectance times its share of the object's light (the same size as with the sun)."""
    import radiometry
    lam = np.asarray(lam_nm, float)
    order = np.argsort(lam)
    weights, flux_total, _ = radiometry.calibrate(
        mag, "V", "johnson", lam_nm=lam[order], T_sun=T_SUN, reflectance=None, D_tel_m=D_CM / 100.0,
        airmass=1.0 / np.cos(np.radians(ZENITH_DEG)), obstruction_frac=OBSTRUCTION, t_exp_s=T_EXP_S, verbose=False)
    gray = np.empty(len(lam))
    gray[order] = flux_total * weights                      # proportional to B(T) * G * lam * dlam
    if sun:
        return gray
    no_sun = gray / radiometry.blackbody_flambda(lam, T_SUN)
    return no_sun * gray.sum() / no_sun.sum()


def divisor(reading, lam_nm, mag=DEFAULT_MAG):
    """The number each slice is divided by, in the order of `lam_nm`."""
    if reading == "solar":
        return solar_divisor(lam_nm)
    if reading == "radiometry":
        return radiometry_divisor(lam_nm, sun=False, mag=mag)
    if reading == "radiometry_sun":
        return radiometry_divisor(lam_nm, sun=True, mag=mag)
    raise ValueError(f"unknown reading {reading!r}; one of {', '.join(READINGS)}")


def to_reflectance(cube, response, snr_floor=1e-3, normalise="max"):
    """cube [slices, y, x] in electrons -> relative reflectance (as recover_reflectance.py).
    Pixels whose summed value is below snr_floor of the brightest are NaN. normalise: 'max' | 'none' | a slice index."""
    refl = cube / response[:, None, None]
    total = cube.sum(axis=0)
    refl[:, total < snr_floor * total.max()] = np.nan
    if normalise == "max":
        refl /= np.nanmax(refl)
    elif isinstance(normalise, (int, np.integer)):
        refl /= np.nanmedian(refl[normalise])
    return refl


def disk_spectrum(cube, response, mask):
    """The spectrum summed over `mask`, divided by the response and scaled to a maximum of 1."""
    s = cube[:, mask].sum(1) / response
    return s / s.max()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(ROOT / "assets/new_input"), help="folder with estimate.fits (and object.fits)")
    ap.add_argument("--out", default=str(ROOT / "outputs/fits_reading"))
    ap.add_argument("--reading", default="radiometry", choices=READINGS[1:])
    ap.add_argument("--mag", type=float, default=DEFAULT_MAG, help="only sets the overall constant")
    ap.add_argument("--normalise", default="max", help="max | none | a slice index")
    ap.add_argument("--snr-floor", type=float, default=1e-3)
    a = ap.parse_args()

    from astropy.io import fits
    norm = a.normalise if a.normalise in ("max", "none") else int(a.normalise)
    lam = wavelength_grid()                                  # red -> blue, as the files
    resp = divisor(a.reading, lam, a.mag)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    hdr["BUNIT"] = "relative reflectance"
    hdr["READING"] = (a.reading, "how the electrons were divided")
    hdr["SUNINC"] = (a.reading == "radiometry_sun", "truth cube assumed to include the sun")
    hdr["MAG"] = (a.mag, "sets only the overall constant")
    for k, w in enumerate(lam):
        hdr[f"WAVE{k}"] = (round(float(w), 3), f"slice {k} wavelength [nm]")

    est = fits.getdata(Path(a.run_dir) / "estimate.fits").astype(float)
    if est.shape[0] != len(lam):
        sys.exit(f"estimate has {est.shape[0]} slices, the grid has {len(lam)}; check the settings at the top of this file")
    fits.writeto(out / "reflectance_estimate.fits", to_reflectance(est, resp, a.snr_floor, norm).astype(np.float32), hdr, overwrite=True)
    print("wrote", out / "reflectance_estimate.fits")
    obj_path = Path(a.run_dir) / "object.fits"
    if obj_path.is_file():
        obj = fits.getdata(obj_path).astype(float)
        refl_obj = to_reflectance(obj, resp, a.snr_floor, norm)
        fits.writeto(out / "reflectance_truth.fits", refl_obj.astype(np.float32), hdr, overwrite=True)
        print("wrote", out / "reflectance_truth.fits")
        refl_est = to_reflectance(est, resp, a.snr_floor, norm)
        m = np.isfinite(refl_est).all(0) & np.isfinite(refl_obj).all(0)
        de, dt = disk_spectrum(est, resp, m), disk_spectrum(obj, resp, m)
        print(f"\n{'lambda':>8} {'est (disk)':>11} {'truth (disk)':>12}")
        for k in np.argsort(lam):
            print(f"{lam[k]:8.1f} {de[k]:11.4f} {dt[k]:12.4f}")


if __name__ == "__main__":
    main()
