"""
Sub-pixel / subset-shape refinement for the subset tracker of
core/dic_processing.py: inverse-compositional Gauss-Newton (IC-GN) with a
FIRST-ORDER (affine) subset shape function and a zero-normalized SSD
criterion, the algorithm used by the standard 2D-DIC packages (Ncorr, VIC-2D
...). It starts from the integer + parabolic estimate of the NCC peak and
refines it.

Why it exists. The plain tracker (cv2.matchTemplate + 3-point parabola on the
correlation peak) suffers from two systematic errors that show up as
VERTICAL / HORIZONTAL STRIPES in the strain maps:

* "peak locking": the parabola fit on an integer-shift correlation map is
  biased towards integer (and half-integer) displacements, by up to ~0.1-0.2
  px, with a period of 1 px of displacement. Displacement varies smoothly over
  the specimen, so the bias oscillates along it; the strain is a DERIVATIVE of
  the displacement, so this oscillation is amplified by the local displacement
  gradient: the stripes follow the iso-displacement lines (vertical where u
  changes with x, horizontal where v changes with y), and get stronger and
  closer together as the strain grows.
* the rigid template cannot follow the deformation of its own subset (a 5 %
  strain over a 21 px subset is a 1 px stretch), which adds a position-
  dependent error that depends on the speckle in each subset.

The affine IC-GN fit interpolates the deformed image with a cubic kernel and
adjusts displacement AND its first gradient inside each subset, which removes
both. All functions are pure numpy/OpenCV, no Qt.
"""
from __future__ import annotations

import functools

import cv2
import numpy as np

_FLAT_EPS = 1e-6


@functools.lru_cache(maxsize=16)
def _local_grid(size: int):
    c = (size - 1) / 2.0
    j = np.arange(size, dtype=np.float32) - c
    xi, eta = np.meshgrid(j, j)          # xi varies along columns (x), eta along rows (y)
    return xi.ravel(), eta.ravel()   # shared between all subsets of a size: do not modify


class IcgnTemplate:
    """Everything about one reference subset that does not change from frame
    to frame (so it is computed once): the normalized template, its steepest-
    descent images and the inverse Hessian."""
    __slots__ = ("x0", "y0", "size", "t_norm", "sd", "h_inv", "xi", "eta")

    def __init__(self, x0, y0, size, t_norm, sd, h_inv, xi, eta):
        self.x0, self.y0, self.size = x0, y0, size
        self.t_norm, self.sd, self.h_inv, self.xi, self.eta = t_norm, sd, h_inv, xi, eta


def prepare_template(ref_f: np.ndarray, x0: int, y0: int, size: int) -> IcgnTemplate | None:
    """Reference subset at array index (x0, y0) (top-left), `size` x `size`
    pixels, from the float32 reference frame. None when the subset is flat
    (no texture to track) or numerically degenerate."""
    h, w = ref_f.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + size > w or y0 + size > h:
        return None
    # gradient from a 1-px wider patch (replicated at the frame border)
    pad = ref_f[max(y0 - 1, 0):min(y0 + size + 1, h), max(x0 - 1, 0):min(x0 + size + 1, w)]
    top, left = (1 if y0 >= 1 else 0), (1 if x0 >= 1 else 0)
    pad = cv2.copyMakeBorder(pad, 1 - top, 1 - (1 if y0 + size + 1 <= h else 0),
                             1 - left, 1 - (1 if x0 + size + 1 <= w else 0), cv2.BORDER_REPLICATE)
    gx = 0.5 * (pad[1:-1, 2:] - pad[1:-1, :-2])
    gy = 0.5 * (pad[2:, 1:-1] - pad[:-2, 1:-1])
    patch = ref_f[y0:y0 + size, x0:x0 + size].astype(np.float32)
    tm = patch - patch.mean()
    dt = float(np.sqrt((tm ** 2).sum()))
    if not np.isfinite(dt) or dt < _FLAT_EPS * size * size + 1e-9:
        return None
    t_norm = (tm / dt).ravel().astype(np.float32)
    gx = (gx / dt).ravel().astype(np.float32)
    gy = (gy / dt).ravel().astype(np.float32)
    xi, eta = _local_grid(size)
    sd64 = np.column_stack([gx, gx * xi, gx * eta, gy, gy * xi, gy * eta]).astype(np.float64)
    hess = sd64.T @ sd64
    sd = sd64.astype(np.float32)          # float32 storage: half the memory of the cached templates
    try:
        if np.linalg.cond(hess) > 1e12:
            return None
        h_inv = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return None
    return IcgnTemplate(x0, y0, size, t_norm, sd, h_inv, xi, eta)


def refine_icgn(frame_f: np.ndarray, tpl: IcgnTemplate, u0: float, v0: float,
                max_iter: int = 20, tol: float = 1e-3, max_shift_px: float = 2.0):
    """Refines the displacement (u0, v0) of one subset. Returns (u, v, zncc,
    ok): `zncc` is the zero-normalized cross-correlation coefficient of the
    refined match (1 = perfect); `ok` False means the refinement did not
    converge to something plausible, in which case the caller should keep its
    initial estimate (u, v are then (u0, v0))."""
    size, n = tpl.size, tpl.size * tpl.size
    cxi = tpl.x0 + (size - 1) / 2.0
    cyi = tpl.y0 + (size - 1) / 2.0
    xi, eta = tpl.xi.astype(np.float64), tpl.eta.astype(np.float64)
    sd_t = tpl.sd.T.astype(np.float64)
    # homogeneous affine matrix of the warp (reference -> current), local coordinates
    a = np.array([[1.0, 0.0, u0], [0.0, 1.0, v0], [0.0, 0.0, 1.0]])
    last_c = None
    for _ in range(max_iter):
        mx = (cxi + a[0, 0] * xi + a[0, 1] * eta + a[0, 2]).astype(np.float32).reshape(size, size)
        my = (cyi + a[1, 0] * xi + a[1, 1] * eta + a[1, 2]).astype(np.float32).reshape(size, size)
        cur = cv2.remap(frame_f, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).ravel()
        fm = cur - cur.mean()
        df = float(np.sqrt(np.dot(fm, fm)))
        if df < _FLAT_EPS * n + 1e-9:
            return u0, v0, float("nan"), False
        r = fm / df - tpl.t_norm
        last_c = float(np.dot(r, r))
        dp = tpl.h_inv @ (sd_t @ r.astype(np.float64))           # [du, dux, duy, dv, dvx, dvy]
        d_a = np.array([[1.0 + dp[1], dp[2], dp[0]], [dp[4], 1.0 + dp[5], dp[3]], [0.0, 0.0, 1.0]])
        try:
            a = a @ np.linalg.inv(d_a)
        except np.linalg.LinAlgError:
            return u0, v0, float("nan"), False
        if not np.all(np.isfinite(a)):
            return u0, v0, float("nan"), False
        if abs(dp[0]) < tol and abs(dp[3]) < tol and max(abs(dp[1]), abs(dp[2]), abs(dp[4]), abs(dp[5])) < tol * 0.1:
            break
    u, v = float(a[0, 2]), float(a[1, 2])
    # final criterion at the converged warp
    mx = (cxi + a[0, 0] * xi + a[0, 1] * eta + a[0, 2]).astype(np.float32).reshape(size, size)
    my = (cyi + a[1, 0] * xi + a[1, 1] * eta + a[1, 2]).astype(np.float32).reshape(size, size)
    cur = cv2.remap(frame_f, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).ravel()
    fm = cur - cur.mean()
    df = float(np.sqrt(np.dot(fm, fm)))
    if df < _FLAT_EPS * n + 1e-9:
        return u0, v0, float("nan"), False
    r = fm / df - tpl.t_norm
    zncc = 1.0 - 0.5 * float(np.dot(r, r))
    if (not np.isfinite(zncc)) or abs(u - u0) > max_shift_px or abs(v - v0) > max_shift_px:
        return u0, v0, float("nan"), False
    return u, v, zncc, True


# ---------------------------------------------------------------------------
# Batched version: many subsets of the same size refined together. Same maths
# as `refine_icgn`, but one cv2.remap call and a few numpy einsums per
# iteration for the whole batch instead of Python-level work per subset
# (an order of magnitude faster, and the numpy / OpenCV calls release the GIL
# so batches can be spread over threads).
# ---------------------------------------------------------------------------
class IcgnBatch:
    """Stacked templates of `m` subsets (all `size` x `size`)."""
    __slots__ = ("size", "m", "cx", "cy", "t_norm", "sd", "h_inv", "xi", "eta")

    def __init__(self, tpls):
        self.size = tpls[0].size
        self.m = len(tpls)
        self.cx = np.array([t.x0 + (t.size - 1) / 2.0 for t in tpls])
        self.cy = np.array([t.y0 + (t.size - 1) / 2.0 for t in tpls])
        self.t_norm = np.stack([t.t_norm for t in tpls])           # (m, n) float32
        self.sd = np.stack([t.sd for t in tpls])                   # (m, n, 6) float32
        self.h_inv = np.stack([t.h_inv for t in tpls])             # (m, 6, 6) float64
        self.xi = tpls[0].xi.astype(np.float64)
        self.eta = tpls[0].eta.astype(np.float64)


def _sample_batch(frame_f, b: IcgnBatch, rows, a):
    """Current-frame values of the warped subsets `rows` (a: (k,3,3))."""
    size, k = b.size, len(rows)
    mx = (b.cx[rows, None] + a[:, 0, 0, None] * b.xi + a[:, 0, 1, None] * b.eta + a[:, 0, 2, None])
    my = (b.cy[rows, None] + a[:, 1, 0, None] * b.xi + a[:, 1, 1, None] * b.eta + a[:, 1, 2, None])
    cur = cv2.remap(frame_f, mx.astype(np.float32).reshape(k * size, size),
                    my.astype(np.float32).reshape(k * size, size),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cur.reshape(k, size * size)


def _normalize_batch(cur, n):
    fm = cur - cur.mean(axis=1, keepdims=True)
    df = np.sqrt(np.einsum("kn,kn->k", fm, fm))
    flat = ~(df >= _FLAT_EPS * n + 1e-9)
    df = np.where(flat, 1.0, df)
    return fm / df[:, None], flat


def refine_batch(frame_f: np.ndarray, b: IcgnBatch, rows, u0, v0,
                 max_iter: int = 20, tol: float = 1e-3, max_shift_px: float = 2.0):
    """Refines the subsets `rows` (indices into the batch) starting from the
    template-centre displacements (u0, v0) (arrays, one per row). Returns
    (u, v, zncc, ok) arrays; where `ok` is False, (u, v) = (u0, v0) and the
    caller should keep its own estimate."""
    rows = np.asarray(rows, dtype=int)
    u0 = np.asarray(u0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    k, n = len(rows), b.size * b.size
    a = np.zeros((k, 3, 3))
    a[:, 0, 0] = a[:, 1, 1] = a[:, 2, 2] = 1.0
    a[:, 0, 2], a[:, 1, 2] = u0, v0
    alive = np.ones(k, dtype=bool)
    conv = np.zeros(k, dtype=bool)
    for _ in range(max_iter):
        pos = np.nonzero(alive & ~conv)[0]
        if pos.size == 0:
            break
        rr = rows[pos]
        r, flat = _normalize_batch(_sample_batch(frame_f, b, rr, a[pos]), n)
        r = r - b.t_norm[rr]
        grad = np.einsum("knj,kn->kj", b.sd[rr], r.astype(np.float32)).astype(np.float64)
        dp = np.einsum("kij,kj->ki", b.h_inv[rr], grad)
        det = (1.0 + dp[:, 1]) * (1.0 + dp[:, 5]) - dp[:, 2] * dp[:, 4]
        bad = flat | ~np.isfinite(dp).all(axis=1) | ~(np.abs(det) > 1e-6)
        dp[bad] = 0.0
        d_a = np.zeros((pos.size, 3, 3))
        d_a[:, 0, 0], d_a[:, 0, 1], d_a[:, 0, 2] = 1.0 + dp[:, 1], dp[:, 2], dp[:, 0]
        d_a[:, 1, 0], d_a[:, 1, 1], d_a[:, 1, 2] = dp[:, 4], 1.0 + dp[:, 5], dp[:, 3]
        d_a[:, 2, 2] = 1.0
        a[pos] = a[pos] @ np.linalg.inv(d_a)
        alive[pos[bad]] = False
        small = ((np.abs(dp[:, 0]) < tol) & (np.abs(dp[:, 3]) < tol)
                 & (np.abs(dp[:, [1, 2, 4, 5]]).max(axis=1) < tol * 0.1))
        conv[pos[small & ~bad]] = True
    alive &= np.isfinite(a).reshape(k, -1).all(axis=1)
    zncc = np.full(k, np.nan)
    pos = np.nonzero(alive)[0]
    if pos.size:
        rr = rows[pos]
        r, flat = _normalize_batch(_sample_batch(frame_f, b, rr, a[pos]), n)
        r = r - b.t_norm[rr]
        zncc[pos] = 1.0 - 0.5 * np.einsum("kn,kn->k", r, r)
        alive[pos[flat]] = False
    u, v = a[:, 0, 2].copy(), a[:, 1, 2].copy()
    ok = alive & np.isfinite(zncc) & (np.abs(u - u0) <= max_shift_px) & (np.abs(v - v0) <= max_shift_px)
    u[~ok], v[~ok], zncc[~ok] = u0[~ok], v0[~ok], np.nan
    return u, v, zncc, ok
