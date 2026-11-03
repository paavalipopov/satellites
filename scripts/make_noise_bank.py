"""Cut patches with NO object out of a real restored cube, to use as the generator's noise bank.

The restoration's background is not white Gaussian noise (in estimate.fits 99% of the pixels are exactly
0, with faint specks around the object), so real examples are the best way to teach the model what
"nothing here" looks like. This reads a 10-band .fits cube like notebook 06 (V / (E.dlam), blue to red),
divides it by the satellite's brightest value (so the patches are in units of "the object's peak", like
the training cubes), and saves patches that are at least --margin pixels from anything bright.

    python scripts/make_noise_bank.py                                      # assets/new_input/estimate.fits
    python scripts/make_noise_bank.py --fits /path/to/another_restoration.fits --out assets/noise_bank.npy

Use it in training with   EXTRA="+data.params.sim.noise_bank=assets/noise_bank.npy" sbatch ...
(optionally  +data.params.sim.noise_bank_log10_scale=[-1.0,0.5]  to vary its strength).

CAUTION: a bank cut from estimate.fits makes estimate.fits a training source for the noise (never for
the object, which is excluded). Score such a model on a DIFFERENT restoration, or say so when you report it.
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BANDS_10 = np.array([400.0, 426.667, 457.143, 492.308, 533.333, 581.818, 640.0, 711.111, 800.0, 914.286])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", default=str(ROOT / "assets/new_input/estimate.fits"))
    ap.add_argument("--out", default=str(ROOT / "assets/noise_bank.npy"))
    ap.add_argument("--size", type=int, default=15, help="patch size in px (the generator's image_size)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--margin", type=int, default=10, help="stay this many px away from any pixel above --bright")
    ap.add_argument("--bright", type=float, default=0.01, help="'bright' = above this fraction of the image maximum")
    ap.add_argument("--min-nonzero", type=float, default=0.01, help="drop patches with fewer non-zero pixels than this share")
    args = ap.parse_args()

    import pvlib
    from astropy.io import fits
    from scipy import ndimage

    solar = pvlib.spectrum.get_reference_spectra(standard="ASTM G173-03")["extraterrestrial"]
    divisor = np.interp(BANDS_10, solar.index, solar.values) * BANDS_10 ** 2 / 6400
    cube = fits.getdata(args.fits).astype(float)[::-1]
    if cube.shape[0] != 10:
        raise SystemExit(f"{args.fits}: {cube.shape[0]} bands; this script is for the 10-band cube")
    reading = cube / divisor[:, None, None]
    total = reading.sum(0)
    bright = total > args.bright * total.max()
    sat = reading[:, bright].max()                      # the object's peak, as "satellite max = 1"
    near = ndimage.binary_dilation(bright, iterations=args.margin)
    reading = reading / sat

    patches = []
    s = args.size
    for y in range(0, reading.shape[1] - s + 1, args.stride):
        for x in range(0, reading.shape[2] - s + 1, args.stride):
            if near[y:y + s, x:x + s].any():
                continue
            p = reading[:, y:y + s, x:x + s]
            if (p.sum(0) != 0).mean() >= args.min_nonzero:
                patches.append(p)
    if not patches:
        raise SystemExit("no patch has any signal; lower --min-nonzero, or give a file with restoration artifacts")
    bank = np.stack(patches).astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, bank)
    print(f"{len(bank)} patches {bank.shape[1:]} saved to {args.out}; "
          f"max {bank.max():.2e} of the object's peak, mean {bank.mean():.2e}")


if __name__ == "__main__":
    main()
