"""Displacement components in the LOCAL frame of the nearest crack.

A branching specimen has several cracks running in different directions, so
"the displacement perpendicular to the crack" is not one image axis: it is
u . n with n the normal of THE CRACK CLOSEST to each point. This module builds
that field on the DIC grid: every grid node is assigned to the nearest point of
any given crack path, and u, v are projected on that point's tangent t (the
direction of crack growth, smoothed over `tangent_window_px`) and normal
n = t rotated by +90 degrees in the image coordinates (x right, y down), i.e.

    u_t =  u * tx + v * ty
    u_n = -u * ty + v * tx

Across an open crack u_n changes sign (the faces move apart): on one side the
points go along +n, on the other along -n, so the field shows the opening as a
red/blue pair of bands along every crack, each in its own frame.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from core.crack_tip_tracking import path_tangents


def nearest_crack_frame(xs, ys, paths, tangent_window_px: float = 16.0, return_beyond: bool = False,
                        return_foot: bool = False):
    """For every node of the grid (xs along columns, ys along rows) the tangent
    (tx, ty) of the nearest crack path point, the index of that crack in
    `paths` and the distance to it (px).

    `paths` is a list of (positions (n, 2) in px, arc_length (n,)) -- each
    walked in the direction of growth. Returns (tx, ty, owner, dist), each of
    shape (len(ys), len(xs)); tx = ty = NaN and owner = -1 when `paths` is
    empty or holds no usable point. With `return_beyond=True` a fifth array is
    appended: True where the nearest point is the START or the END of its path
    and the node lies beyond it (past the end of the crack, not beside it). With
    `return_foot=True` the nearest path point itself, an array (ny, nx, 2), is
    appended after that."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    shape = (len(ys), len(xs))
    pts, tans, own, end_dir = [], [], [], []
    for k, (pos, arc) in enumerate(paths):
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2 or len(pos) < 2:
            continue
        tan = path_tangents(pos, np.asarray(arc, dtype=float), window_px=tangent_window_px)
        good = np.isfinite(pos).all(axis=1) & np.isfinite(tan).all(axis=1)
        pts.append(pos[good]); tans.append(tan[good]); own.append(np.full(int(good.sum()), k))
        # outward direction at the two ends of this path (0 elsewhere): -t at the start, +t at the end
        outward = np.zeros((int(good.sum()), 2))
        if outward.shape[0] >= 2:
            outward[0] = -tan[good][0]
            outward[-1] = tan[good][-1]
        end_dir.append(outward)
    if not pts or sum(len(p) for p in pts) == 0:
        nan = np.full(shape, np.nan)
        out = (nan, nan.copy(), np.full(shape, -1), nan.copy())
        if return_beyond:
            out = out + (np.zeros(shape, dtype=bool),)
        if return_foot:
            out = out + (np.full(shape + (2,), np.nan),)
        return out
    pts = np.vstack(pts); tans = np.vstack(tans); own = np.concatenate(own); end_dir = np.vstack(end_dir)
    X, Y = np.meshgrid(xs, ys)
    q = np.column_stack([X.ravel(), Y.ravel()])
    dist, idx = cKDTree(pts).query(q)
    out = (tans[idx, 0].reshape(shape), tans[idx, 1].reshape(shape), own[idx].reshape(shape), dist.reshape(shape))
    if return_beyond:
        beyond = np.einsum("ij,ij->i", q - pts[idx], end_dir[idx]) > 0.0          # only ever true at a path end
        out = out + (beyond.reshape(shape),)
    if return_foot:
        out = out + (pts[idx].reshape(shape + (2,)),)
    return out


def local_frame_displacement(u, v, xs, ys, paths, tangent_window_px: float = 16.0,
                             band_inner_px: float | None = None, band_outer_px: float | None = None):
    """(u_n, u_t): displacement normal / tangential to the nearest crack, same
    shape as `u` (NaN where u or v is NaN, or where there is no crack).

    `band_outer_px` (None = the whole field) keeps ONLY the pixels within that
    distance of a crack path, on either side of it -- the local measure along
    the crack faces; `band_inner_px` additionally drops the pixels closer than
    that (the face itself, where the subset DIC is blurred/decorrelated). Pixels
    beyond the START or END of a crack path are dropped too, as soon as a band
    is requested: the band follows the crack, it does not cap its tips."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    banded = band_outer_px is not None or band_inner_px is not None
    tx, ty, _own, dist, beyond = nearest_crack_frame(xs, ys, paths, tangent_window_px, return_beyond=True)
    u_n, u_t = -u * ty + v * tx, u * tx + v * ty
    if banded:
        drop = beyond.copy()
        if band_outer_px is not None:
            drop |= dist > band_outer_px
        if band_inner_px:
            drop |= dist < band_inner_px
        u_n = np.where(drop, np.nan, u_n)
        u_t = np.where(drop, np.nan, u_t)
    return u_n, u_t


def local_jump_fields(u, v, xs, ys, paths, tangent_window_px: float = 16.0,
                      band_inner_px: float | None = None, band_outer_px: float | None = 10.0):
    """The LOCAL discontinuity of the displacement across each crack, mapped on
    the pixels of a band along it -- the quantity that tells whether a crack is
    open, free of the bulk displacement (rigid translation, wave) that makes u_n
    itself unreadable (it is the same on both faces and dwarfs the opening).

    For a node p beside a crack, with q the nearest path point, t and n the local
    tangent / normal and d = (p - q) . n its signed distance, the MIRROR node is
    p' = p - 2 d n (the same position along the crack, the other face, the same
    distance). The jump between the two is  J = w(+n side) - w(-n side)  with w =
    (u, v) and the same value is given to p AND p':

        opening  = J . n   > 0 when the faces move apart, < 0 when they overlap
        sliding  = J . t   relative tangential displacement of the faces

    both independent of the direction the path is walked. Nodes outside the
    band, beyond a crack end, whose mirror falls outside the field or on a NaN,
    are NaN. Returns (opening, sliding, owner) -- `owner` is the index in `paths`
    of the crack each node belongs to (-1 outside), to normalise per crack."""
    from scipy.interpolate import RegularGridInterpolator
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    tx, ty, owner, dist, beyond, foot = nearest_crack_frame(
        xs, ys, paths, tangent_window_px, return_beyond=True, return_foot=True)
    X, Y = np.meshgrid(xs, ys)
    nx, ny = -ty, tx
    d = (X - foot[..., 0]) * nx + (Y - foot[..., 1]) * ny
    mx, my = X - 2.0 * d * nx, Y - 2.0 * d * ny
    ok = np.isfinite(d)
    q = np.column_stack([np.where(ok, my, 0.0).ravel(), np.where(ok, mx, 0.0).ravel()])
    shape = u.shape
    um = RegularGridInterpolator((ys, xs), u, bounds_error=False, fill_value=np.nan)(q).reshape(shape)
    vm = RegularGridInterpolator((ys, xs), v, bounds_error=False, fill_value=np.nan)(q).reshape(shape)
    sgn = np.where(d >= 0, 1.0, -1.0)                      # +n side minus -n side, whichever pixel we stand on
    ju, jv = sgn * (u - um), sgn * (v - vm)
    opening = ju * nx + jv * ny
    sliding = ju * tx + jv * ty
    drop = beyond | ~ok
    if band_outer_px is not None:
        drop |= dist > band_outer_px
    if band_inner_px:
        drop |= dist < band_inner_px
    opening = np.where(drop, np.nan, opening)
    sliding = np.where(drop, np.nan, sliding)
    return opening, sliding, np.where(drop, -1, owner)


def normalise_per_crack(values, owner, mode_scale: float | None = None, per_crack_max: bool = False,
                        percentile: float = 98.0):
    """Divide `values` by a fixed scale (e.g. the detection threshold) or, with
    `per_crack_max`, by each crack's own `percentile` of the positive values --
    so every crack reads 0..1 whatever its absolute opening. Nodes without a
    positive reference are left NaN."""
    values = np.asarray(values, dtype=float)
    if per_crack_max:
        out = np.full(values.shape, np.nan)
        for k in np.unique(owner[owner >= 0]):
            m = (owner == k) & np.isfinite(values)
            pos = values[m][values[m] > 0]
            if pos.size:
                out[m] = values[m] / float(np.percentile(pos, percentile))
        return out
    if mode_scale and mode_scale > 0:
        return values / float(mode_scale)
    return values
