"""Stage 1a — geometry, and the rasteriser that turns it into a material mask.

Written with numpy only.  The original used `trimesh` for three primitives and
`scipy.spatial.transform` for one rotation matrix; neither is worth a dependency.

The important thing to understand here: the rasteriser writes a *material index*
per pixel, not a brightness.  There is no lighting model anywhere — no surface
normals, no cos(incidence), no phase angle, no shadowing between components.
A panel seen edge-on and a panel seen face-on produce identical values.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import SimConfig

Mesh = tuple[np.ndarray, np.ndarray]          # (vertices [V,3], faces [F,3])


# ----------------------------------------------------------------------
# primitives
def box(extents) -> Mesh:
    ex, ey, ez = (np.asarray(extents, float) / 2.0)
    v = np.array([[-ex, -ey, -ez], [+ex, -ey, -ez], [+ex, +ey, -ez], [-ex, +ey, -ez],
                  [-ex, -ey, +ez], [+ex, -ey, +ez], [+ex, +ey, +ez], [-ex, +ey, +ez]])
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                  [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                  [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return v, f


def cylinder(radius: float, height: float, sections: int = 20) -> Mesh:
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ring = np.stack([radius * np.cos(a), radius * np.sin(a)], 1)
    bot = np.hstack([ring, np.full((sections, 1), -height / 2)])
    top = np.hstack([ring, np.full((sections, 1), +height / 2)])
    v = np.vstack([bot, top, [[0, 0, -height / 2]], [[0, 0, +height / 2]]])
    cb, ct = 2 * sections, 2 * sections + 1
    f = []
    for i in range(sections):
        j = (i + 1) % sections
        f += [[i, j, sections + j], [i, sections + j, sections + i]]   # wall
        f += [[cb, j, i], [ct, sections + i, sections + j]]            # caps
    return v, np.array(f)


def icosphere(subdivisions: int = 2, radius: float = 1.0) -> Mesh:
    t = (1.0 + 5 ** 0.5) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(subdivisions):
        v, f = _subdivide(v, f)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * radius
    return v, f


def _subdivide(v: np.ndarray, f: np.ndarray) -> Mesh:
    v = list(map(tuple, v))
    cache: dict[tuple[int, int], int] = {}

    def mid(a: int, b: int) -> int:
        k = (min(a, b), max(a, b))
        if k not in cache:
            cache[k] = len(v)
            v.append(tuple((np.array(v[a]) + np.array(v[b])) / 2.0))
        return cache[k]

    nf = []
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(v, float), np.array(nf)


def translate(mesh: Mesh, t) -> Mesh:
    v, f = mesh
    return v + np.asarray(t, float), f


def concat(*meshes: Mesh) -> Mesh:
    vs, fs, off = [], [], 0
    for v, f in meshes:
        vs.append(v); fs.append(f + off); off += len(v)
    return np.vstack(vs), np.vstack(fs)


# ----------------------------------------------------------------------
# camera
def euler_xyz(angles_deg) -> np.ndarray:
    """Extrinsic x-y-z rotation — matches scipy `Rotation.from_euler('xyz', degrees=True)`.

    Lowercase 'xyz' in scipy means rotations about the FIXED axes applied in the
    order x, y, z, which composes as Rz @ Ry @ Rx (not Rx @ Ry @ Rz).
    """
    ax, ay, az = np.radians(angles_deg)
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def intrinsics(fov_deg: float, n: int) -> np.ndarray:
    f = (0.5 * n) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0, n / 2], [0, f, n / 2], [0, 0, 1]])


# ----------------------------------------------------------------------
def rasterize(components, cfg: SimConfig, angles_deg, image_size: int | None = None):
    """Z-buffered rasterisation of (mesh, material_id) pairs -> uint8 mask.

    components : sequence of ((vertices, faces), material_id)
    returns    : mask [n, n] uint8, values 0 (background) or a material id
    """
    n = image_size or cfg.image_size
    zbuf = np.full((n, n), np.inf)
    mask = np.zeros((n, n), np.uint8)

    R = euler_xyz(angles_deg)
    K = intrinsics(cfg.fov_deg, n)
    cam = np.array([0.0, 0.0, cfg.camera_height])

    for (verts, faces), mat_id in components:
        V = verts @ R.T + cam                       # into camera frame
        hom = V @ K.T
        uv = hom[:, :2] / hom[:, 2:3]               # pixel coordinates
        z = V[:, 2]
        for tri in faces:
            if cfg.rasterizer == "cv2":
                _fill_cv2(mask, zbuf, uv[tri], z[tri], int(mat_id))
            else:
                _fill(mask, zbuf, uv[tri], z[tri], int(mat_id))
    return mask


def _fill_cv2(mask, zbuf, uv, z, mat_id):
    """The original's coverage rule: truncate vertices to int32 and let
    `cv2.fillConvexPoly` decide.  Kept because it is the only reason the
    0.01-thick solar panels ever appeared.  Depth is one value per face."""
    n = mask.shape[0]
    pts = uv.astype(np.int32)
    poly = np.zeros((n, n), np.uint8)
    cv2.fillConvexPoly(poly, pts, 1)
    depth = float(np.mean(z))
    upd = (poly == 1) & (depth < zbuf)
    if upd.any():
        zbuf[upd] = depth
        mask[upd] = mat_id


def _fill(mask, zbuf, uv, z, mat_id):
    """Exact pixel-centre coverage, with perspective-correct depth."""
    n = mask.shape[0]
    x0 = max(int(np.floor(uv[:, 0].min())), 0)
    x1 = min(int(np.ceil(uv[:, 0].max())) + 1, n)
    y0 = max(int(np.floor(uv[:, 1].min())), 0)
    y1 = min(int(np.ceil(uv[:, 1].max())) + 1, n)
    if x1 <= x0 or y1 <= y0:
        return

    (ax, ay), (bx, by), (cx, cy) = uv
    det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(det) < 1e-12:
        return

    ys, xs = np.mgrid[y0:y1, x0:x1]
    px, py = xs + 0.5, ys + 0.5
    w1 = ((px - ax) * (cy - ay) - (cx - ax) * (py - ay)) / det
    w2 = ((bx - ax) * (py - ay) - (px - ax) * (by - ay)) / det
    w0 = 1.0 - w1 - w2
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return

    # perspective-correct: 1/z is linear in screen space
    inv = w0 / z[0] + w1 / z[1] + w2 / z[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(inv > 0, 1.0 / inv, np.inf)

    sub = zbuf[y0:y1, x0:x1]
    upd = inside & (depth < sub)
    if upd.any():
        sub[upd] = depth[upd]
        mask[y0:y1, x0:x1][upd] = mat_id
