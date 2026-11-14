"""Steps after the model, at test time: how the runs are combined, the outline taken from the data, and cleanup.

Nothing here is trained; each step is a switch, so that its effect can be measured on its own.

  vote      the runs of an experiment are combined: a pixel is an object where at least `threshold` of the runs say
            so (0.5 = the notebooks' rule), and gets the material most runs give it.
  outline   "model"          the model's own outline.
            "mask@<frac>"    labels outside the outline found in the DATA are dropped. The data outline is every pixel
                             brighter than <frac> of the image's brightest (and clearly above the noise), see
                             auto_scale.object_blobs. estimate.fits is exactly 0 away from the object, so this is a
                             cheap outline; a smaller <frac> keeps fainter pixels (and more specks).
            "fill@<frac>"    the same, and a pixel inside that outline that the model called background gets the
                             material of the nearest labelled pixel of the same object (nothing to copy = stays 0).
            "...+faint"      (mask@0.01+faint, fill@0.01+faint) the data outline also holds the faint objects of the
                             second search (auto_scale.find_faint), e.g. the companion; use with per_object_faint.
  cleanup   "off"
            "tidy"           objects of fewer than 3 labelled pixels are removed; background pixels enclosed by an
                             object get the nearest material. With a "+faint" outline, labels on a faint object found by
                             the second search (the companion: 1-2 labelled pixels) are kept, not removed as specks.
            "tidy+smooth"    also: each labelled pixel takes the material most common in its 3x3 neighbourhood.
"""
import numpy as np
from scipy import ndimage

import auto_scale
from src.data_generation import N_CLASSES

VOTES = (0.3, 0.5, 0.7)
OUTLINES = ("model", "mask@0.01", "fill@0.01", "mask@0.001", "fill@0.001", "mask@0.01+faint", "fill@0.01+faint")
CLEANUPS = ("off", "tidy", "tidy+smooth")


def vote(preds, threshold=0.5):
    """preds [runs, H, W] -> labels [H, W]."""
    counts = (preds[..., None] == np.arange(1, N_CLASSES)).sum(0)
    objectness = (preds > 0).mean(0)
    return np.where(objectness >= threshold, counts.argmax(-1) + 1, 0)


def _fill(labels, region):
    """Inside `region` (bool), pixels labelled 0 take the label of the nearest labelled pixel of the same region."""
    have = (labels > 0) & region
    if not have.any():
        return labels
    _, (iy, ix) = ndimage.distance_transform_edt(~have, return_indices=True)
    out = labels.copy()
    todo = region & (labels == 0)
    out[todo] = labels[iy, ix][todo]
    return out


def apply_outline(labels, cube, outline):
    """labels [H, W]; cube [bands, H, W]: the image the model was given (any overall scale)."""
    if outline == "model":
        return labels
    kind, _, frac = outline.partition("@")
    frac, plus, extra = frac.partition("+")
    if kind not in ("mask", "fill") or not frac or (plus and extra != "faint"):
        raise ValueError(f"unknown outline {outline!r}; one of {', '.join(OUTLINES)}")
    cores, _ = auto_scale.object_blobs(cube, frac=float(frac), faint=bool(plus))
    out = np.where(cores > 0, labels, 0)
    if kind == "fill":
        for k in range(1, int(cores.max()) + 1):
            out = _fill(out, cores == k)
    return out


def _majority(labels):
    """Each labelled pixel takes the most common material in its 3x3 neighbourhood (ties keep its own)."""
    present = [c for c in np.unique(labels) if c > 0]
    if len(present) < 2:
        return labels
    score = np.stack([ndimage.uniform_filter((labels == c).astype(float), size=3, mode="constant") * 9
                      + 0.01 * (labels == c) for c in present])
    smooth = np.array(present)[score.argmax(0)]
    return np.where(labels > 0, smooth, 0)


def cleanup(labels, mode, keep=None):
    """keep [H, W] bool: labelled groups touching these pixels are never removed as specks (the faint objects)."""
    if mode == "off":
        return labels
    if mode not in CLEANUPS:
        raise ValueError(f"unknown cleanup {mode!r}; one of {', '.join(CLEANUPS)}")
    out = labels.copy()
    blobs, n = ndimage.label(out > 0, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(out > 0, blobs, range(1, n + 1))
        for k in np.flatnonzero(np.asarray(sizes) < auto_scale.MIN_PX) + 1:
            if keep is not None and (keep & (blobs == k)).any():
                continue                                   # a faint object the second search found, not a speck
            out[blobs == k] = 0
    holes = ndimage.binary_fill_holes(out > 0) & (out == 0)
    if holes.any():
        out = _fill(out, holes | (out > 0))
    if mode == "tidy+smooth":
        out = _majority(out)
    return out


def faint_only(cube, outline):
    """The pixels of the objects that only the second search found (none unless the outline ends in "+faint")."""
    if not outline.endswith("+faint"):
        return None
    frac = float(outline.split("@")[1].split("+")[0])
    with_faint, _ = auto_scale.object_blobs(cube, frac=frac, faint=True)
    bright, _ = auto_scale.object_blobs(cube, frac=frac)
    return (with_faint > 0) & (bright == 0)


def finish(labels, cube, outline="model", clean="off"):
    """The data outline first, then the cleanup (which keeps the faint objects of a "+faint" outline)."""
    return cleanup(apply_outline(labels, cube, outline), clean, keep=faint_only(cube, outline))
