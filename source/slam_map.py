"""
SLAM map of the maze from the recorded drives: pose-graph least squares + a free-space (wall) map.

Coordinates: metres, yaw in radians counter-clockwise, the spawn is the origin facing yaw 0. Every recorded frame of every
drive (IDLE frames too: the robot has momentum and keeps moving through them) gets a pose (x, y, yaw).

Measurements (each a residual / its sigma, solved together by robust least squares):
  1. MOTION MODEL    frame i-1 -> i moves (forward, left, turn) as learned for the last 3 actions (slam_calibrate.py).
  2. VISUAL ODOMETRY consecutive kept frames: their rotation from the essential matrix of their keypoint matches
                     (simulator camera fx=92 at 320x240); trusted more the more matches it has.
  3. LOOP CLOSURES   pairs of frames that show the same place (route_agent's verified links + ones found here):
                     relative rotation from the essential matrix, position from the keypoint size change.
  4. SPAWN           every drive starts at the spawn.
  5. MAZE GRID       the maze is a grid of square cells and the spawn faces along it, so while driving straight
                     (3 FORWARD frames in a row) the heading is within a few degrees of 0/90/180/270. This is what keeps
                     the heading from drifting (measured on local mazes: median 2.5 degrees off the grid on straight runs).
Solved in 3 stages: headings (linear, after a drift-resetting forward pass), then positions given the headings (linear),
then everything together (nonlinear, optional). Afterwards the map is used to look for MORE loop closures (frames the map
puts together are checked by their pictures) and solved again.
Walls: free space = within FREE_R of anywhere a drive went. A straight walk is allowed only if it stays inside it.

Offline check against the simulator's truth for a local maze (prints errors, draws the map over the true walls):
    uv run python source/slam_map.py 22
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np


def _env(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return type(default)(v)
    except ValueError:
        return default


FX = _env("SLAM_FX", 92.0)                     # camera focal length in px at 320 px image width (simulator camera)
VO_SKIPS = _env("SLAM_VO_SKIPS", 2)           # visual odometry between frames k and k+1 ... k+VO_SKIPS (redundancy: a wrong one gets outvoted)
SIG_LINK_M = _env("SLAM_SIG_LINK_M", 0.08)     # how many metres two linked frames may be off the link's measurement
SIG_LINK_DX_DEG = _env("SLAM_SIG_LINK_DX_DEG", 8.0)  # degrees a link rotation may be off when only the picture shift is known
DEG_PER_DX = _env("SLAM_DEG_PER_DX", 0.455)    # picture shift (px) -> degrees (fallback when the essential matrix fails)
M_PER_LS = _env("SLAM_M_PER_LS", 0.15)         # keypoint size change (log2) -> metres along the view
GRID = _env("SLAM_GRID", 0)                    # 1 = use the maze-grid heading prior (5. above). Off: tested on maze 22 it snapped drifted headings to the WRONG grid direction (map error 0.84 m -> 4.5 m)
SIG_GRID_DEG = _env("SLAM_SIG_GRID_DEG", 4.0)  # ...how many degrees a straight-run heading may be off the grid
SNAP_MAX_DEG = _env("SLAM_SNAP_MAX_DEG", 30.0) # ...forward pass: only snap when the estimate is this close to a grid direction
LOSS_SCALE = _env("SLAM_LOSS_SCALE", 2.0)      # residuals (in sigmas) above this count as outliers
ROBUST = os.environ.get("SLAM_ROBUST", "cauchy")  # outlier weighting of the linear stages: "cauchy" (a wrong measurement is ignored almost entirely) or "huber"
ITERS = _env("SLAM_ITERS", 30)                 # nonlinear polish iterations (0 = linear stages only)
CLOSE_ROUNDS = _env("SLAM_CLOSE_ROUNDS", 3)    # rounds of "find frames the map puts together, check their pictures, add closures, solve again"
CLOSE_M = _env("SLAM_CLOSE_M", 0.25)           # ...frames at most this far apart in the current map are checked
CLOSE_DEG = _env("SLAM_CLOSE_DEG", 35.0)       # ...and facing at most this many degrees apart
CLOSE_GAP = _env("SLAM_CLOSE_GAP", 15)         # ...and (same drive) at least this many frames apart in time
CLOSE_MIN_INL = _env("SLAM_CLOSE_MIN_INL", 30) # ...pictures must share this many keypoint matches
CLOSE_YAW_DEG = _env("SLAM_CLOSE_YAW_DEG", 10.0)  # ...and their measured rotation must agree with the map this well
CLOSE_MAX = _env("SLAM_CLOSE_MAX", 12000)      # ...at most this many candidate pairs checked per round
FREE_R = _env("SLAM_FREE_R", 0.07)             # free space around every pose a drive visited (m); maze cells are 0.2 m
SEG_STEP = 0.02                                # sampling of a straight walk when testing it against the walls
QUARTER = math.pi / 2

_MODEL = None


def motion_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = json.loads(Path(__file__).with_name("motion_model.json").read_text())
    return _MODEL


def motion(ctx):
    """(forward, left, turn, sf, sl, st) for the last actions ctx (oldest first), falling back to shorter contexts."""
    table = motion_model()["table"]
    for L in range(len(ctx), 0, -1):
        e = table.get("|".join(ctx[-L:]))
        if e is not None:
            return e["f"], e["l"], e["t"], e["sf"], e["sl"], e["st"]
    return 0.0, 0.0, 0.0, 0.05, 0.05, 0.2


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def yaw_sigma(inliers: int) -> float:
    """How far a measured rotation may be off, by its number of agreeing matches (measured against local truth: with few
    matches -- blank walls -- the rotation is often wrong)."""
    return math.radians(0.6 if inliers >= 50 else 1.0 if inliers >= 30 else 1.5 if inliers >= 20 else 3.0 if inliers >= 10 else 8.0)


def _bearings(P, width):
    s = width / 320.0
    f = np.c_[(P[:, 0] - width / 2) / (FX * s), (P[:, 1] - 0.75 * width / 2) / (FX * s), np.ones(len(P))]
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def _planar_cost(fa, fb, th, ph, thr):
    """Truncated epipolar cost of every (yaw th, translation direction ph) for planar motion: E = [t]x Ry(th)."""
    th = th[:, None, None]
    ph = ph[None, :, None]
    c, s = np.cos(th), np.sin(th)
    rx, rz = c * fa[:, 0] + s * fa[:, 2], -s * fa[:, 0] + c * fa[:, 2]
    ry = fa[:, 1]
    tx, tz = np.sin(ph), np.cos(ph)
    nx, ny, nz = -tz * ry, tz * rx - tx * rz, tx * ry
    err = np.abs(nx * fb[:, 0] + ny * fb[:, 1] + nz * fb[:, 2]) / (np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-12)
    return np.minimum(err, thr), err


def rel_yaw(A, B, width=320.0, thr=0.004):
    """(CCW yaw change from view A to view B, agreeing matches) from matched points, or None.
    The robot moves on a flat floor, so the camera motion has only 2 unknowns: the turn and the direction it moved in.
    Both are found by a robust grid search (works for turning on the spot too, where the essential matrix breaks down);
    measured on local mazes: median error 0.35 degrees per frame pair."""
    if len(A) < 8:
        return None
    fa, fb = _bearings(A, width), _bearings(B, width)
    th = np.radians(np.arange(-50.0, 50.01, 1.0))
    ph = np.radians(np.arange(0.0, 360.0, 12.0))
    C = (_planar_cost(fa, fb, th, ph, thr)[0] ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(C), C.shape)
    th2 = th[i] + np.radians(np.arange(-1.0, 1.01, 0.05))
    ph2 = ph[j] + np.radians(np.arange(-12.0, 12.1, 1.5))
    C2, err = _planar_cost(fa, fb, th2, ph2, thr)
    C2 = (C2 ** 2).sum(-1)
    i2, j2 = np.unravel_index(np.argmin(C2), C2.shape)
    n = int((err[i2, j2] < thr).sum())
    if n < 6:
        return None
    return float(th2[i2]), n


class Drives:
    """All recorded frames (IDLE included) of every drive, and where the agent's kept frames sit among them."""

    def __init__(self, frames):
        self.acts, self.drive, self.first = [], [], []
        self.kept = np.zeros(len(frames), int)
        by_dir = {}
        for k, f in enumerate(frames):
            by_dir.setdefault(Path(f.path).parent, []).append(k)
        for d, (directory, ks) in enumerate(sorted(by_dir.items(), key=lambda kv: frames[kv[1][0]].trajectory)):
            recs = json.loads((directory / "data_info.json").read_text())
            base = len(self.acts)
            self.first.append(base)
            index = {r["image"]: base + j for j, r in enumerate(recs)}
            for r in recs:
                self.acts.append(r["action"][0] if len(r["action"]) == 1 else "IDLE")
                self.drive.append(d)
            for k in ks:
                self.kept[k] = index[Path(frames[k].path).name]
        self.n = len(self.acts)
        self.drive = np.array(self.drive)
        self.straight = np.array([i >= 2 and self.drive[i - 2] == self.drive[i]
                                  and self.acts[i] == self.acts[i - 1] == self.acts[i - 2] == "FORWARD" for i in range(self.n)])

    def context(self, i, L=3):
        d = self.drive[i]
        return tuple(self.acts[j] if j >= self.first[d] else "START" for j in range(i - L + 1, i + 1))


def dead_reckon(D: Drives):
    P = np.zeros((D.n, 3))
    for i in range(D.n):
        prev = np.zeros(3) if i in D.first else P[i - 1]
        f, l, t, *_ = motion(D.context(i))
        c, s = math.cos(prev[2]), math.sin(prev[2])
        P[i] = prev[0] + f * c - l * s, prev[1] + f * s + l * c, prev[2] + t
    return P


# ---------------------------------------------------------------------------------------------- linear solvers
def _irls(rows, cols, vals, rhs, sig, n, rounds=8):
    """Robust linear least squares: minimise sum huber((A x - rhs) / sig) by re-weighting. Returns x, residuals (sigmas)."""
    from scipy.sparse import coo_matrix, identity
    from scipy.sparse.linalg import spsolve
    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), n)).tocsr()
    w = 1.0 / sig ** 2
    reg = 1e-9 * identity(n, format="csc")
    for it in range(rounds):
        x = spsolve((A.T @ A.multiply(w[:, None]).tocsr()).tocsc() + reg, A.T @ (w * rhs))
        r = (A @ x - rhs) / sig
        if ROBUST == "cauchy" and it >= 1:
            w = 1.0 / (1.0 + (r / LOSS_SCALE) ** 2) / sig ** 2
        else:
            w = np.minimum(1.0, LOSS_SCALE / np.maximum(np.abs(r), 1e-9)) / sig ** 2
    return x, r


class _Eqs:
    """Collects sparse linear equations sum(coef * x[col]) = rhs (+- sig)."""

    def __init__(self):
        self.rows, self.cols, self.vals, self.rhs, self.sig, self.e = [], [], [], [], [], 0

    def add(self, terms, rhs, sig):
        m = len(rhs)
        k = np.arange(m) + self.e
        for col, coef in terms:
            self.rows.append(k); self.cols.append(np.asarray(col)); self.vals.append(np.broadcast_to(coef, (m,)).astype(float))
        self.rhs.append(np.asarray(rhs, float)); self.sig.append(np.broadcast_to(sig, (m,)).astype(float))
        self.e += m

    def solve(self, n):
        c = np.concatenate
        return _irls(c(self.rows), c(self.cols), c(self.vals), c(self.rhs), c(self.sig), n)


def _forward_headings(D, mm_t, vo):
    """Drift-resetting forward pass: add up the turns (visual odometry where it agrees with the motion model), and on
    every straight run snap the heading to the nearest grid direction. Every drive starts at the spawn heading (0)."""
    inc = mm_t.copy()
    sig_m = np.array([motion(D.context(i))[5] for i in range(D.n)])
    for a, b, y, s in vo:
        msum = inc[a + 1:b + 1].sum()
        if abs(wrap(y - msum)) < 3 * math.sqrt(np.sum(sig_m[a + 1:b + 1] ** 2) + s * s):
            inc[b] += y - msum
    H = np.zeros(D.n)
    snap = math.radians(SNAP_MAX_DEG)
    for i in range(D.n):
        H[i] = (0.0 if i in D.first else H[i - 1]) + inc[i]
        if GRID and D.straight[i]:
            q = round(H[i] / QUARTER) * QUARTER
            if abs(H[i] - q) < snap:
                H[i] = q
    return H


def solve_headings(D, mm, vo, links, rounds=3):
    """Headings alone: linear once each relative rotation's whole turns (2 pi k) and each straight run's grid direction
    are fixed from the current estimate (the forward pass keeps that estimate far inside +-45 degrees)."""
    mt = np.array([m[2] for m in mm])
    H = _forward_headings(D, mt, vo)
    idx = np.arange(D.n)
    firsts = np.array(D.first)
    mi = np.setdiff1d(idx, firsts)
    st = np.array([m[5] for m in mm])
    for _ in range(rounds):
        q = _Eqs()
        q.add([(mi, 1.0), (mi - 1, -1.0)], mt[mi], st[mi])                  # motion model
        q.add([(firsts, 1.0)], mt[firsts], st[firsts])                       # first move from the spawn (heading 0)
        rel =[(a, b, y, s) for a, b, y, s in vo] + [(a, b, y, s) for a, b, y, s, _ in links]
        if rel:
            a, b, y, s = (np.array(v) for v in zip(*rel))
            cur = H[b] - H[a]
            q.add([(b, 1.0), (a, -1.0)], cur + wrap(y - cur), s)             # y + 2 pi k, k from the current estimate
        if GRID:
            g = idx[D.straight]
            q.add([(g, 1.0)], np.round(H[g] / QUARTER) * QUARTER, math.radians(SIG_GRID_DEG))
        H, _ = q.solve(D.n)
    return H


def solve_positions(D, H, mm, links):
    """With the headings known, every position measurement is LINEAR in (X, Y)."""
    n = D.n
    idx = np.arange(n)
    firsts = np.array(D.first)
    mi = np.setdiff1d(idx, firsts)
    f, l, sf, sl = (np.array([m[k] for m in mm]) for k in (0, 1, 3, 4))
    p = mi - 1
    c, s = np.cos(H[p]), np.sin(H[p])
    q = _Eqs()
    # along the previous heading: (X_i - X_p) c + (Y_i - Y_p) s = f ; sideways: -(X_i - X_p) s + (Y_i - Y_p) c = l
    q.add([(mi, c), (p, -c), (n + mi, s), (n + p, -s)], f[mi], sf[mi])
    q.add([(mi, -s), (p, s), (n + mi, c), (n + p, -c)], l[mi], sl[mi])
    q.add([(firsts, 1.0)], f[firsts], sf[firsts])                             # spawn at the origin facing yaw 0
    q.add([(n + firsts, 1.0)], l[firsts], sl[firsts])
    if links:
        a, b, _, _, fwd = (np.array(v) for v in zip(*links))
        q.add([(b, 1.0), (a, -1.0)], fwd * np.cos(H[a]), SIG_LINK_M)
        q.add([(n + b, 1.0), (n + a, -1.0)], fwd * np.sin(H[a]), SIG_LINK_M)
    x, _ = q.solve(2 * n)
    return x[:n], x[n:]


def polish(D, P, mm, vo, links):
    """Everything together (nonlinear, robust soft-L1), starting from the linear solution."""
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix
    n = D.n
    firsts = np.array(D.first)
    mi = np.setdiff1d(np.arange(n), firsts)
    M_ = np.array(mm)
    rel = [(a, b, y, s) for a, b, y, s in vo] + [(a, b, y, s) for a, b, y, s, _ in links]
    ra, rb, ry, rs = (np.array(v) for v in zip(*rel)) if rel else (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
    la, lb, lf = (np.array([L[k] for L in links]) for k in (0, 1, 4)) if links else (np.zeros(0, int),) * 2 + (np.zeros(0),)
    g = np.where(D.straight)[0] if GRID else np.zeros(0, int)
    sg = math.radians(SIG_GRID_DEG)

    def res(x):
        X, Y, H = x[:n], x[n:2 * n], x[2 * n:]
        p = mi - 1
        c, s = np.cos(H[p]), np.sin(H[p])
        dX, dY = X[mi] - X[p], Y[mi] - Y[p]
        out = [(dX * c + dY * s - M_[mi, 0]) / M_[mi, 3], (-dX * s + dY * c - M_[mi, 1]) / M_[mi, 4],
               wrap(H[mi] - H[p] - M_[mi, 2]) / M_[mi, 5],
               (X[firsts] - M_[firsts, 0]) / M_[firsts, 3], (Y[firsts] - M_[firsts, 1]) / M_[firsts, 4],
               wrap(H[firsts] - M_[firsts, 2]) / M_[firsts, 5],
               wrap(H[rb] - H[ra] - ry) / rs,
               (X[lb] - X[la] - lf * np.cos(H[la])) / SIG_LINK_M, (Y[lb] - Y[la] - lf * np.sin(H[la])) / SIG_LINK_M,
               wrap(4 * H[g]) / 4 / sg]
        return np.concatenate(out)

    rows, cols, r = [], [], 0

    def block(u, v):
        nonlocal r
        k = np.arange(len(u)) + r
        for w in (u, v):
            for off in (0, n, 2 * n):
                rows.append(k); cols.append(np.asarray(w) + off)
        r += len(u)

    for u, v in [(mi - 1, mi)] * 3 + [(firsts, firsts)] * 3 + [(ra, rb)] + [(la, lb)] * 2 + [(g, g)]:
        block(u, v)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    S = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(r, 3 * n)).tocsr()
    S.data[:] = 1
    sol = least_squares(res, np.concatenate([P[:, 0], P[:, 1], P[:, 2]]), jac_sparsity=S, loss="soft_l1",
                        f_scale=LOSS_SCALE, method="trf", tr_solver="lsmr", max_nfev=ITERS, x_scale="jac")
    return np.stack([sol.x[:n], sol.x[n:2 * n], sol.x[2 * n:]], 1)


def solve(D: Drives, vo, links, log=print):
    """vo: [(raw a, raw b, yaw, sigma)]; links: [(raw a, raw b, yaw, sigma, forward m)]. Returns poses (n, 3)."""
    t0 = time.time()
    mm = [motion(D.context(i)) for i in range(D.n)]
    H = solve_headings(D, mm, vo, links)
    X, Y = solve_positions(D, H, mm, links)
    P = np.stack([X, Y, H], 1)
    if ITERS > 0:
        P = polish(D, P, mm, vo, links)
    log(f"slam: {D.n} poses, {len(vo)} visual-odometry rotations, {len(links)} loop closures"
        f"{', maze-grid heading prior on ' + str(int(D.straight.sum())) + ' straight-run frames' if GRID else ''} "
        f"-- solved in {time.time() - t0:.0f}s")
    return P


def _link(D, T, feats, a, b, width, dx=None):
    """A loop closure between kept frames a and b from their pictures: (raw a, raw b, yaw, sigma, forward m), or None."""
    A, B = T._inlier_points(feats(a), feats(b))
    if len(A) == 0:
        return None
    r = rel_yaw(A, B, width)
    _, dx_m, ls = T._scale_stats(A, B)
    if r is not None:
        y, s = r[0], yaw_sigma(r[1])
    else:
        y, s = -math.radians(DEG_PER_DX * (dx if dx is not None else dx_m)), math.radians(SIG_LINK_DX_DEG)
    return int(D.kept[a]), int(D.kept[b]), float(y), float(s), float(M_PER_LS * ls), len(A), r is not None


def build(frames, feats, links_raw, log=print, width=320.0):
    """The map from the agent's data. frames: kept frames; feats(k) -> (pts, desc); links_raw {(a, b): (inl, dx)}.
    Returns (Drives, raw poses (n_raw, 3), kept poses (n_kept, 3), [(kept a, kept b)] closures found here)."""
    import tracking_agent as T
    from scipy.spatial import cKDTree
    D = Drives(frames)
    t0 = time.time()
    vo = []
    for skip in range(1, VO_SKIPS + 1):
        for k in range(len(frames) - skip):
            if frames[k].trajectory != frames[k + skip].trajectory:
                continue
            A, B = T._inlier_points(feats(k), feats(k + skip))
            r = rel_yaw(A, B, width) if len(A) else None
            if r is not None:
                vo.append((int(D.kept[k]), int(D.kept[k + skip]), r[0], yaw_sigma(r[1])))
    links = []
    for (a, b), (inl, dx) in links_raw.items():
        L = _link(D, T, feats, a, b, width, dx)
        if L is not None:
            links.append(L[:5])
    log(f"slam: visual odometry on {len(vo)} frame pairs (skips 1..{VO_SKIPS}), {len(links)} known links ({time.time() - t0:.0f}s)")
    P = solve(D, vo, links, log)
    # loop-closure search: frames the map puts together are checked by their pictures; the real ones become closures
    tried = {(min(a, b), max(a, b)) for a, b in links_raw}
    traj = np.array([f.trajectory for f in frames])
    found = []
    for rnd in range(CLOSE_ROUNDS):
        t1 = time.time()
        Q = P[D.kept]
        cand = []
        for a, b in cKDTree(Q[:, :2]).query_pairs(CLOSE_M):
            if (a, b) in tried or (traj[a] == traj[b] and b - a < CLOSE_GAP) \
                    or abs(wrap(Q[b, 2] - Q[a, 2])) > math.radians(CLOSE_DEG):
                continue
            cand.append((float(np.hypot(*(Q[b, :2] - Q[a, :2]))), a, b))
        cand.sort()
        new = 0
        for _, a, b in cand[:CLOSE_MAX]:
            tried.add((a, b))
            L = _link(D, T, feats, a, b, width)
            if L is None or L[5] < CLOSE_MIN_INL or not L[6] or abs(wrap(L[2] - (Q[b, 2] - Q[a, 2]))) > math.radians(CLOSE_YAW_DEG):
                continue
            links.append(L[:5])
            found.append((a, b))
            new += 1
        log(f"slam: closure search {rnd + 1}: {len(cand)} nearby pairs, {new} confirmed by their pictures ({time.time() - t1:.0f}s)")
        if new == 0:
            break
        P = solve(D, vo, links, log)
    return D, P, P[D.kept], found


class FreeSpace:
    """Where the drives went = where you can walk. Anything else is wall (or unexplored: the same for routing)."""

    def __init__(self, P, r=FREE_R):
        from scipy.spatial import cKDTree
        self.P, self.r = P, r
        self.tree = cKDTree(P[:, :2])

    def free(self, pts) -> np.ndarray:
        d, _ = self.tree.query(np.atleast_2d(pts), k=1)
        return d <= self.r

    def clear(self, a, b) -> bool:
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        m = max(2, int(math.ceil(np.hypot(*(b - a)) / SEG_STEP)) + 1)
        t = np.linspace(0.0, 1.0, m)[:, None]
        return bool(self.free(a + t * (b - a)).all())


def draw(P, drive, path_png, free_r=FREE_R, route=None, goal=None, here=None, grid=None, cell=0.2, spawn=(0, 0, 0), scale=120):
    """Top-down picture: drives in colour, route white, goal red, you yellow; true walls grey when grid is given (local
    mazes only), else the free space."""
    import cv2

    def world(Q):
        c, s = math.cos(spawn[2]), math.sin(spawn[2])
        return np.stack([spawn[0] + Q[:, 0] * c - Q[:, 1] * s, spawn[1] + Q[:, 0] * s + Q[:, 1] * c], 1)

    W = world(P)
    lo = W.min(0) - 0.3 if grid is None else np.array([-cell / 2, -cell / 2])
    hi = W.max(0) + 0.3 if grid is None else np.array([(len(grid[0]) - 0.5) * cell, (len(grid) - 0.5) * cell])
    img = np.full((int((hi[1] - lo[1]) * scale), int((hi[0] - lo[0]) * scale), 3), 25, np.uint8)
    px = lambda q: (int((q[0] - lo[0]) * scale), int((q[1] - lo[1]) * scale))
    if grid is not None:
        for r_, row in enumerate(grid):
            for c_, ch in enumerate(row):
                if ch == "#":
                    cv2.rectangle(img, px((c_ * cell - cell / 2, r_ * cell - cell / 2)), px((c_ * cell + cell / 2, r_ * cell + cell / 2)), (90, 90, 90), -1)
    else:
        for q in W[::2]:
            cv2.circle(img, px(q), max(1, int(free_r * scale)), (60, 60, 60), -1)
    cols = [(255, 140, 60), (60, 200, 90), (60, 120, 255), (200, 80, 200)]
    for i in range(len(W) - 1):
        if drive[i] == drive[i + 1]:
            cv2.line(img, px(W[i]), px(W[i + 1]), cols[drive[i] % 4], 1, cv2.LINE_AA)
    if route is not None and len(route) > 1:
        R = world(np.asarray(route, float))
        for a, b in zip(R[:-1], R[1:]):
            cv2.line(img, px(a), px(b), (255, 255, 255), 2, cv2.LINE_AA)
    if goal is not None:
        cv2.circle(img, px(world(np.atleast_2d(goal))[0]), 6, (0, 0, 255), -1)
    if here is not None:
        cv2.circle(img, px(world(np.atleast_2d(here))[0]), 6, (0, 255, 255), -1)
    cv2.circle(img, px(W[0]), 4, (255, 255, 0), -1)
    cv2.imwrite(str(path_png), img)


if __name__ == "__main__":
    import pickle
    import sys

    import vis_nav_sim as sim

    import slam_calibrate as C
    from vlad import load_frames

    seed = int(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("cache") / f"local-{seed}" / "slam_map.png"
    data, cache = Path("data") / f"local-{seed}", Path("cache") / f"local-{seed}"
    frames, _ = load_frames(data)
    n = len(frames)
    raw = json.loads(sorted(cache.glob(f"route_links_v*_{n}_*.json"))[-1].read_text())
    store = pickle.loads((cache / f"rootsift3_feats_{n}.pkl").read_bytes())

    def feats(i):
        p, d = store[str(frames[i].path)]
        return p, d.astype(np.float32)

    D, P, _, found = build(frames, feats, {(a, b): (inl, dx) for a, b, inl, dx in raw})
    # ---- compare with the truth (local simulator only; the agent never sees it) ----
    truth_dir = C.record_truth(seed, Path("cache") / "slam_truth")
    G = np.concatenate([np.array(json.loads((truth_dir / f"traj_{d}" / "poses.json").read_text())) for d in range(len(D.first))])
    w = sim.Simulator(sim.Textures.find(), seed)
    sx, sy, syaw = w.spawn
    c, s = math.cos(syaw), math.sin(syaw)
    grid = np.array([[ch == "#" for ch in row] for row in w.grid])

    def report(name, Q):
        Wx, Wy = sx + Q[:, 0] * c - Q[:, 1] * s, sy + Q[:, 0] * s + Q[:, 1] * c
        e = np.hypot(Wx - G[:, 0], Wy - G[:, 1])
        he = np.degrees(np.abs(wrap(Q[:, 2] + syaw - G[:, 2])))
        rr = np.clip(np.round(Wy / w.cell_size).astype(int), 0, grid.shape[0] - 1)
        cc = np.clip(np.round(Wx / w.cell_size).astype(int), 0, grid.shape[1] - 1)
        print(f"{name:28s} position error median {np.median(e):.2f} m, 90% {np.percentile(e, 90):.2f} m; heading error median "
              f"{np.median(he):.1f} deg; poses in open cells {(~grid[rr, cc]).mean():.0%}")

    report("dead reckoning", dead_reckon(D))
    report("SLAM", P)
    K = D.kept
    if found:
        fa, fb = np.array(found).T
        dd = np.hypot(*(G[K[fa], :2] - G[K[fb], :2]).T)
        print(f"closures found by the map: {len(found)}, truly within 0.3 m: {(dd < 0.3).mean():.0%}")
    # local accuracy = what routing needs: of the frame pairs the map puts within 0.25 m, how many truly are close?
    from scipy.spatial import cKDTree
    Q, GQ = P[K], G[K]
    tr = np.array([f.trajectory for f in frames])
    pairs = np.array(sorted(cKDTree(Q[:, :2]).query_pairs(0.25)))
    if len(pairs):
        a, b = pairs.T
        td = np.hypot(*(GQ[a, :2] - GQ[b, :2]).T)
        same, span = tr[a] == tr[b], np.abs(b - a)
        for name, m in (("same drive, <= 20 frames apart", same & (span <= 20)), ("same drive, 21-60 apart", same & (span > 20) & (span <= 60)),
                        ("same drive, 61-150 apart", same & (span > 60) & (span <= 150)), ("same drive, > 150 apart", same & (span > 150)),
                        ("different drives", ~same)):
            if m.any():
                print(f"  map-close pairs, {name:32s}: {m.sum():6d}, truly within 0.30 m {(td[m] < 0.3).mean():5.0%}")
    draw(P, D.drive, out, grid=w.grid, cell=w.cell_size, spawn=w.spawn)
    print("map picture:", out)
