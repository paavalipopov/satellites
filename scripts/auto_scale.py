"""Automatic brightness scaling in FRONT of the model: Pavel's "set the brightest value to 1", with no person involved.

Nothing is learned and the model is not touched. The image is divided by a number (or by one number per object)
that is worked out from the image itself, then given to the model as it is. The modes:

  none        the image as it is (what the original model gets today)
  single      one number for the whole image: the brightest value among the object pixels. An image with no object
              in it is not given to the model at all (every pixel is background).
  per_object  every separate object gets its own number and its own run of the model, then the answers are put
              back together. This is what the notebook did by hand for the companion ("satellite removed, own scale").
  single_mean, single_p99, per_object_mean, per_object_p99
              the same with another number: the MEAN or the 99th PERCENTILE of the object's values (all bands) instead
              of its brightest. They match the training of pair_mean* / pair_p99* (src/datasets/hsi.py, copy_scale).
  per_object_faint
              per_object, plus a SECOND search for faint objects (find_faint): the bright objects are set aside with
              `faint_margin` pixels round them (the specks that hug them), and the rest of the image is searched
              again. Its brightest pixel becomes an object if it is at least `faint_ratio` times brighter than every
              other pixel left (outside `faint_radius` of it) and above the noise; repeat until none is. This is
              for the companion of estimate.fits: 1 true pixel, about 0.1% of the satellite's brightest, below the
              1% of the first search, but 2.5x brighter than any speck more than 3 px from the satellite.

What counts as an object (nothing here looks at the truth):

  * the band-summed image is the "brightness" map; a pixel belongs to an object if its brightness is above
    `frac` (1%) of the image's brightest pixel AND above `noise_k` times the noise level of the image
    (1.48 x the median absolute deviation of the brightness map; 0 for estimate.fits, whose background is exactly 0);
  * touching object pixels (8 neighbours) form one blob; blobs of fewer than `min_px` pixels are specks and ignored;
  * in per_object mode each blob also owns the pixels within `margin` pixels of it, the faint skirt (the nearest blob
    wins). Pixels owned by nobody are background. The model sees an object's territory only: everything else is set to 0.

Limits, on purpose left visible: two objects that TOUCH are one blob and get one scale; an object dimmer than
`noise_k` times the noise is not found; the scale is the maximum of the blob, so one hot pixel inside an object sets it.
`q` (default 1.0 = the maximum, as in the notebook) can be lowered, e.g. 0.99, to resist that.

    from auto_scale import predict_scaled
    labels = predict_scaled(lambda x: experiment.predict(x), cube, "per_object", n_runs=len(experiment.models))
"""
import numpy as np
from scipy import ndimage

MODES = ("none", "single", "single_mean", "single_p99", "per_object", "per_object_mean", "per_object_p99", "per_object_faint")
FRAC, NOISE_K, MIN_PX, MARGIN = 0.01, 5.0, 3, 3
FAINT_MARGIN, FAINT_RATIO, FAINT_RADIUS, FAINT_GROW, FAINT_MAX = 3, 2.0, 2, 0.25, 3


def find_faint(total, cores, sigma, noise_k=NOISE_K, margin=FAINT_MARGIN, ratio=FAINT_RATIO, radius=FAINT_RADIUS,
               grow=FAINT_GROW, max_objects=FAINT_MAX):
    """The second search: faint objects away from the bright ones. total [H, W] (band-summed), cores [H, W] int
    (the bright objects, 0 = none), sigma = the noise level of `total`. Returns cores with the faint objects added
    as new ids. A faint object = its brightest pixel (at least `ratio` x every other pixel left more than `radius`
    from it, and above noise_k x sigma) and the touching pixels at least `grow` x that pixel. The level a candidate
    must beat only goes up from one candidate to the next, so the specks left after the companion cannot pass by
    being alone."""
    cores = cores.copy()
    speck = 0.0                                            # the brightest speck seen so far: a candidate must beat it
    for _ in range(max_objects):
        if cores.any():
            near = ndimage.distance_transform_edt(cores == 0) <= margin        # the objects and their skirt
        else:
            near = np.zeros(total.shape, bool)
        left = np.where(near, -np.inf, total)
        y, x = np.unravel_index(int(np.argmax(left)), left.shape)
        top = float(left[y, x])
        if not np.isfinite(top) or top <= max(noise_k * sigma, 0.0):
            break
        yy, xx = np.ogrid[:total.shape[0], :total.shape[1]]
        others = np.where((yy - y) ** 2 + (xx - x) ** 2 > radius ** 2, left, -np.inf)
        second = float(others.max())
        if np.isfinite(second):
            speck = max(speck, second)                     # never lowered: once the companion is out, specks compete with it
        if top < ratio * speck:
            break                                          # not clearly brighter than what is left: a speck
        region, _ = ndimage.label((left >= grow * top) & ~near, structure=np.ones((3, 3)))
        cores[region == region[y, x]] = int(cores.max()) + 1
    return cores


def object_blobs(cube, frac=FRAC, noise_k=NOISE_K, min_px=MIN_PX, margin=MARGIN, faint=False):
    """cube [bands, H, W] -> (cores, territory), two int maps [H, W]: 0 = nobody, k = object k (k = 1..n).
    `cores` are the object pixels; `territory` adds the skirt of `margin` pixels (the nearest object wins).
    faint=True adds the second search for faint objects away from the bright ones (find_faint)."""
    cube = np.asarray(cube, float)
    total = cube.sum(0)
    zeros = np.zeros(total.shape, int)
    peak = float(total.max())
    if not peak > 0:
        return zeros, zeros.copy()
    sigma = 1.4826 * float(np.median(np.abs(total - np.median(total))))     # 0 when the background is exactly 0
    core = total > max(frac * peak, noise_k * sigma)
    labelled, n = ndimage.label(core, structure=np.ones((3, 3)))
    sizes = ndimage.sum(core, labelled, range(1, n + 1)) if n else []
    cores = zeros.copy()
    k = 0
    for old, size in enumerate(sizes, 1):
        if size >= min_px:
            k += 1
            cores[labelled == old] = k
    if faint and cores.any():                              # only next to a bright object: an empty window stays empty
        cores = find_faint(total, cores, sigma, noise_k=noise_k)
    if not cores.any():
        return cores, cores.copy()
    dist, (iy, ix) = ndimage.distance_transform_edt(cores == 0, return_indices=True)
    territory = np.where(dist <= margin, cores[iy, ix], 0)
    return cores, territory


def plan(cube, mode, q=1.0, **kw):
    """The runs the model has to do: a list of (territory [H, W] bool, scale). An empty list = nothing to find."""
    cube = np.asarray(cube, float)
    everywhere = np.ones(cube.shape[1:], bool)
    if mode == "none":
        return [(everywhere, 1.0)]
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {', '.join(MODES)}")
    base = mode[:-len("_faint")] if mode.endswith("_faint") else mode
    cores, territory = object_blobs(cube, faint=base != mode, **kw)
    n = int(cores.max())
    if n == 0:
        return []
    kind = "single" if base.startswith("single") else "per_object"
    which = "" if base in ("single", "per_object") else base.split("_")[-1]

    def stat(pixels):
        values = cube[:, pixels]
        if which == "mean":
            return float(values.mean())
        return float(np.quantile(values, 0.99 if which == "p99" else q))

    if kind == "single":
        return [(everywhere, stat(cores > 0))]
    return [(territory == k, stat(cores == k)) for k in range(1, n + 1)]


def predict_scaled(predict, cube, mode, n_runs, **kw):
    """Labels [n_runs, H, W] for one image cube [bands, H, W], scaled as `mode` says.
    predict: a function from a float32 torch tensor [1, bands, H, W] to labels [n_runs, 1, H, W] (numpy or torch)."""
    import torch
    cube = np.asarray(cube, float)
    out = np.zeros((n_runs,) + cube.shape[1:], int)
    for territory, scale in plan(cube, mode, **kw):
        x = np.where(territory[None], cube, 0.0) / max(scale, 1e-12)
        pred = np.asarray(predict(torch.from_numpy(x[None]).float()))[:, 0]
        out[:, territory] = pred[:, territory]
    return out
