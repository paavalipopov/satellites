"""Stage 1b — scene masks.

Three kinds of scene, mixed 30 / 30 / 40 in the original (the satellite is a
minority of the training data, which the manuscript does not mention):

  1. `satellite_scene`  — the 4-part spacecraft, ~13 px span, ~14% foreground
  2. `shapes_scene`     — a few large random geometric regions, ~30% foreground
  3. `patches_scene`    — 12-15 small 1-4 px squares, ~34% foreground, 12 materials

Type 3 is the one that teaches per-pixel spectral discrimination with no shape
prior available, and is the likely reason the "more scenes" model generalises.

Every function takes an explicit `numpy.random.Generator`, so a seed fully
determines the scene.  The original mixed a seeded Generator with the unseeded
global `random` module and was not reproducible.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .geometry import box, concat, cylinder, icosphere, rasterize, translate
from .spectral_library import GROUPS, NAME_TO_ID

SHAPE_TYPES = ("circle", "ring", "rectangle", "line", "ellipse",
               "polygon", "triangle", "diamond", "cross")


# ----------------------------------------------------------------------
def build_satellite(cfg: SimConfig, rng: np.random.Generator, library=None):
    """The 4-part spacecraft.  Returns (components, chosen_material_names).

    Materials are drawn 3-from-bus, 1-from-antenna, 1-from-wings — each part
    only ever wears materials from its own group.  That is a strong prior: the
    model can partly infer material from position rather than from spectrum.
    """
    G = library.groups if library is not None else GROUPS
    comps, chosen = [], {}
    h = 3.0
    half = h / 2.0

    bus = list(rng.choice(G["bus"], 3, replace=False))
    chosen["bus"] = bus
    comps.append((translate(cylinder(0.8, half, 20), [0, 0, -h / 2 + half / 2]), NAME_TO_ID[bus[0]]))
    comps.append((translate(cylinder(0.8, half, 20), [0, 0, +h / 2 - half / 2]), NAME_TO_ID[bus[1]]))
    connectors = concat(translate(box([2.0, 0.3, 0.3]), [+2.0, 0, 0]),
                        translate(box([2.0, 0.3, 0.3]), [-2.0, 0, 0]))
    comps.append((connectors, NAME_TO_ID[bus[2]]))

    ant = str(rng.choice(G["antenna"]))
    chosen["antenna"] = ant
    comps.append((translate(icosphere(2, 0.4), [0, 0, h / 2 + 0.8]), NAME_TO_ID[ant]))

    wing = str(rng.choice(G["wings"]))
    chosen["wings"] = wing
    t = cfg.panel_thickness
    panels = concat(translate(box([4.0, t, 2.0]), [+6.0, 0, 0]),
                    translate(box([4.0, t, 2.0]), [-6.0, 0, 0]))
    comps.append((panels, NAME_TO_ID[wing]))
    return comps, chosen


def build_satellite_v2(cfg: SimConfig, rng: np.random.Generator, library=None):
    """The SECOND spacecraft layout — the original picked between the two with
    equal probability (`idx = np.random.randint(0,2)`), so half of all satellite
    scenes look like this.  It is a substantially bigger object: a wider bus, a
    long boom, and a 3.2-radius dish.

    Note the material draws: the two bus halves come from one
    `choice(..., 2, replace=False)` but the boom is an INDEPENDENT draw, so the
    boom can share a material with a body half.
    """
    G = library.groups if library is not None else GROUPS
    comps, chosen = [], {}
    h_total, r_body = 3.5, 1.25
    h_half = h_total / 2.0

    bus = list(rng.choice(G["bus"], 2, replace=False))
    comps.append((translate(cylinder(r_body, h_half, 24), [0, 0, -h_half / 2]), NAME_TO_ID[bus[0]]))
    comps.append((translate(cylinder(r_body, h_half, 24), [0, 0, +h_half / 2]), NAME_TO_ID[bus[1]]))

    wing = str(rng.choice(G["wings"]))
    px = r_body + 0.12 + 6.0 / 2.0
    panels = concat(translate(box([6.0, cfg.panel_thickness, 3.0]), [+px, 0, 0]),
                    translate(box([6.0, cfg.panel_thickness, 3.0]), [-px, 0, 0]))
    comps.append((panels, NAME_TO_ID[wing]))

    boom_mat = str(rng.choice(G["bus"]))          # independent of `bus`
    comps.append((translate(box([0.15, 0.15, 6.0]), [0, 0, h_half + 3.0]), NAME_TO_ID[boom_mat]))

    ant = str(rng.choice(G["antenna"]))
    dish = translate(cylinder(3.2, 0.03, 64), [0, 0, h_half + 6.0])
    feed = translate(icosphere(2, 0.25), [0, 0, h_half + 5.3])
    comps.append((concat(dish, feed), NAME_TO_ID[ant]))

    # The whole assembly is re-centred on its bounding box.
    lo = np.min([v.min(0) for v, _ in [c[0] for c in comps]], axis=0)
    hi = np.max([v.max(0) for v, _ in [c[0] for c in comps]], axis=0)
    ctr = (lo + hi) / 2.0
    comps = [((v - ctr, f), mid) for (v, f), mid in comps]

    chosen = {"bus": bus, "boom": boom_mat, "antenna": ant, "wings": wing}
    return comps, chosen


def satellite_scene(cfg: SimConfig, rng: np.random.Generator, image_size=None,
                    variant: int | None = None, library=None):
    """The original chose between the two layouts with equal probability."""
    if variant is None:
        variant = int(rng.integers(0, 2))
    builder = build_satellite if variant == 0 else build_satellite_v2
    comps, chosen = builder(cfg, rng, library)
    # Original ranges.  The y (horizontal) term was commented out, so the
    # satellite only ever tips about x and spins about z.
    angles = np.array([float(rng.integers(45, 135)), 0.0, float(rng.integers(-180, 180))])
    mask = rasterize(comps, cfg, angles, image_size)
    return mask, {"kind": "satellite", "variant": variant,
                  "angles_deg": angles, "materials": chosen}


# ----------------------------------------------------------------------
def shapes_scene(cfg: SimConfig, rng: np.random.Generator, max_regions: int = 3,
                 image_size=None, library=None):
    """A few large random regions, each a distinct material.

    `max_regions=3` is the original's value: `shape_simulator` declares it as a
    default and `generator_realtime.simulator_worker` never overrides it, even
    though `random_complex_region_mask` itself defaults to 5.
    """
    n = image_size or cfg.image_size
    pool = library.usable_ids if library is not None else np.arange(1, 47)
    img = np.zeros((n, n), np.uint8)
    used: set[int] = set()
    kinds = []

    def ri(a, b):
        return int(a) if b < a else int(rng.integers(a, b + 1))

    for _ in range(int(rng.integers(1, max_regions + 1))):
        if len(used) >= len(pool):
            break
        lab = int(rng.choice(pool))
        while lab in used:
            lab = int(rng.choice(pool))
        used.add(lab)
        kind = str(rng.choice(SHAPE_TYPES))
        kinds.append(kind)
        _draw_shape(img, kind, lab, n, ri, rng, cfg)

    return img, {"kind": "shapes", "shapes": kinds}


def _draw_shape(img, kind, lab, n, ri, rng, cfg: SimConfig):
    import cv2
    if kind == "circle":
        cv2.circle(img, (ri(0, n - 1), ri(0, n - 1)), ri(2, max(2, n // 6)), lab, -1)
    elif kind == "ring":
        cx, cy = ri(0, n - 1), ri(0, n - 1)
        outer = ri(5, max(5, n // 5)); inner = ri(1, max(1, outer - 2))
        cv2.circle(img, (cx, cy), outer, lab, -1)
        cv2.circle(img, (cx, cy), inner, 0, -1)
    elif kind == "rectangle":
        w, h = ri(2, n // 4), ri(2, n // 4)
        pts = cv2.boxPoints(((ri(w, n - w), ri(h, n - h)), (w, h), ri(0, 180)))
        cv2.fillPoly(img, [pts.astype(np.int32)], lab)
    elif kind == "line":
        cx, cy = ri(0, n - 1), ri(0, n - 1)
        L = ri(4, n // 2); a = np.deg2rad(ri(0, 180))
        dx, dy = int(L * np.cos(a)), int(L * np.sin(a))
        cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), lab, 2)
    elif kind == "ellipse":
        cv2.ellipse(img, (ri(0, n - 1), ri(0, n - 1)),
                    (ri(2, n // 5), ri(2, n // 5)), ri(0, 180), 0, 360, lab, -1)
    elif kind == "polygon":
        k = ri(4, 8)
        pts = np.column_stack([rng.integers(0, n, k), rng.integers(0, n, k)]).astype(np.int32)
        cv2.fillPoly(img, [_rot(pts, (n // 2, n // 2), ri(0, 180))], lab)
    elif kind == "triangle":
        # The original drew only TWO points here, so `fillPoly` produced a
        # degenerate line — every "triangle" in the original training set is a
        # line.  `legacy_triangle_bug` reproduces that.
        k = 2 if cfg.legacy_triangle_bug else 3
        pts = np.column_stack([rng.integers(0, n, k), rng.integers(0, n, k)]).astype(np.int32)
        c = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        cv2.fillPoly(img, [_rot(pts, c, ri(0, 180))], lab)
    elif kind == "diamond":
        cx, cy = ri(0, n - 1), ri(0, n - 1); s = ri(2, max(2, n // 6))
        pts = np.array([[cx, cy - s], [cx + s, cy], [cx, cy + s], [cx - s, cy]], np.int32)
        cv2.fillPoly(img, [_rot(pts, (cx, cy), ri(0, 180))], lab)
    elif kind == "cross":
        cx, cy = ri(0, n - 1), ri(0, n - 1)
        s = ri(4, max(4, n // 6)); th = max(1, s // 4)
        cr = np.zeros((n, n), np.uint8)
        cv2.rectangle(cr, (cx - th, cy - s), (cx + th, cy + s), 255, -1)
        cv2.rectangle(cr, (cx - s, cy - th), (cx + s, cy + th), 255, -1)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), ri(0, 180), 1.0)
        img[cv2.warpAffine(cr, M, (n, n)) > 0] = lab


def _rot(pts, centre, deg):
    import cv2
    M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), deg, 1.0)
    return cv2.transform(pts.reshape(-1, 1, 2).astype(np.float32), M).astype(np.int32)


# ----------------------------------------------------------------------
def patches_scene(cfg: SimConfig, rng: np.random.Generator,
                  patch_sizes=(1, 2, 3, 4), n_patches=(12, 15), image_size=None,
                  library=None):
    """Small non-overlapping squares, each a distinct material."""
    n = image_size or cfg.image_size
    mask = np.zeros((n, n), np.uint8)
    pool = (library.usable_ids if library is not None else np.arange(1, 47)).astype(np.uint8)
    k = int(rng.integers(n_patches[0], n_patches[1] + 1))
    labels = rng.choice(pool, size=min(k, len(pool)), replace=False)

    for lab in labels:
        for _ in range(80):
            s = int(rng.choice(patch_sizes))
            if s > n:
                continue
            y0 = int(rng.integers(0, n - s + 1)); x0 = int(rng.integers(0, n - s + 1))
            region = mask[y0:y0 + s, x0:x0 + s]
            if np.all(region == 0):
                # The original wrote `np.random.randint(1,47)` here instead of
                # `lab`, throwing away its own uniqueness guarantee and using
                # the unseeded global RNG.  `legacy_patch_label_bug` reproduces it.
                region[:] = (int(np.random.randint(1, 47))
                             if cfg.legacy_patch_label_bug else int(lab))
                break
    return mask, {"kind": "patches", "n_patches": k}


# ----------------------------------------------------------------------
SCENE_FNS = (satellite_scene, shapes_scene, patches_scene)


def random_scene(cfg: SimConfig, rng: np.random.Generator, image_size=None, library=None):
    """Draw a scene type according to `cfg.scene_probs` and build it."""
    idx = int(rng.choice(3, p=cfg.scene_probs))
    return SCENE_FNS[idx](cfg, rng, image_size=image_size, library=library)
