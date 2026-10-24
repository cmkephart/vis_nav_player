"""
Phase B: the last metre.

Place recognition (the baseline's VLAD) says which exploration frame we are near. This module
takes over from there. It matches the live frame against the four goal photos, recovers the
relative camera pose with an essential matrix, and turns that into moves:

    far (parallax large) -> turn toward the goal's bearing, then drive in
    near (parallax small) -> turn until view 0 is dead ahead, then check in

Conventions (all angles in degrees):
    yaw_left       > 0 means the goal camera is turned LEFT of ours: turn LEFT to match it.
    bearing_right  > 0 means the goal position is to the RIGHT of straight ahead.
    forward        > 0 means the goal position is ahead of us, < 0 behind us.
    parallax       median pixel residual after cancelling the rotation: ~0 when we stand
                   on the goal position, whatever our heading.

Nothing here needs the baseline: give it intrinsics, the goal views, and frames.
Every number in Params marked GUESS must be tuned on --local runs.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from vis_nav_sdk import Action

# Heading of each goal view relative to view 0, left positive: front, left, back, right.
VIEW_HEADING = (0.0, 90.0, 180.0, -90.0)


def wrap(deg: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class Params:
    ratio: float = 0.8              # Lowe ratio test
    min_matches: int = 12           # ratio-test survivors needed to try a pose
    min_inliers: int = 15           # below this the estimate is ignored (module returns None)
    checkin_inliers: int = 25       # GUESS: stricter bar for the check-in itself
    ransac_px: float = 1.0
    yaw_tol: float = 6.0            # GUESS: deg; the top heading band is 10
    parallax_tol: float = 4.0       # GUESS: px; depends on wall distance, tune it
    bearing_tol: float = 20.0       # GUESS: deg of heading error tolerated while driving
    drive_burst: int = 1            # ticks per FORWARD/BACKWARD request
    turn_deg_per_tick: float | None = None   # unknown until measured; None = one tick per turn
    panic_steps: int = 3            # check in anyway when this few steps are left
    engage_inliers: int = 30        # GUESS: phase B engages by itself at this many inliers


@dataclass
class Estimate:
    view: int
    matches: int
    inliers: int
    yaw_left: float        # toward the matched view's direction
    yaw_front: float       # toward view 0, whichever view matched
    bearing_right: float
    forward: float
    parallax: float


def rotation_only(p_cur: np.ndarray, p_goal: np.ndarray, K: np.ndarray, px: float = 2.0):
    """Pure-rotation fit (H = K R K^-1). Stable when we are on the goal position, where the
    essential matrix is degenerate. Returns ``(yaw_left, parallax, inliers)`` or None."""
    if len(p_cur) < 8:
        return None
    H, mask = cv2.findHomography(p_cur, p_goal, cv2.RANSAC, px)
    if H is None:
        return None
    U, _, Vt = np.linalg.svd(np.linalg.inv(K) @ H @ K)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    keep = mask.ravel() > 0
    if keep.sum() < 8:
        return None
    moved = cv2.perspectiveTransform(p_cur[keep].reshape(-1, 1, 2), K @ R @ np.linalg.inv(K))
    parallax = float(np.median(np.linalg.norm(moved.reshape(-1, 2) - p_goal[keep], axis=1)))
    return math.degrees(math.atan2(R[0, 2], R[2, 2])), parallax, int(keep.sum())


def pose_from_points(
    p_cur: np.ndarray, p_goal: np.ndarray, K: np.ndarray, ransac_px: float = 1.0, rot_px: float = 0.0
):
    """Relative pose of the goal camera as seen from the current one.

    ``p_cur`` and ``p_goal`` are (N, 2) matched pixel coordinates. Returns
    ``(yaw_left, bearing_right, forward, parallax, inliers)`` or None. If a pure rotation already
    explains the matches to within ``rot_px`` pixels we are on the goal position: the yaw then
    comes from the rotation-only fit and bearing/forward are NaN (no direction to give).
    """
    if len(p_cur) < 8:
        return None
    p_cur = np.ascontiguousarray(p_cur, dtype=np.float64)
    p_goal = np.ascontiguousarray(p_goal, dtype=np.float64)
    spin = rotation_only(p_cur, p_goal, K)
    if spin is not None and spin[1] <= rot_px:
        return spin[0], math.nan, math.nan, spin[1], spin[2]
    E, mask = cv2.findEssentialMat(
        p_cur, p_goal, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_px
    )
    if E is None or E.shape[0] < 3:
        return None
    E = E[:3]
    inliers, R, t, mask = cv2.recoverPose(E, p_cur, p_goal, K, mask=mask)
    if inliers < 8:
        return None

    # x_goal = R x_cur + t. A point straight ahead of us lands at x = sin(yaw) in the goal
    # camera, so positive yaw means the goal camera looks further left than we do.
    yaw_left = math.degrees(math.atan2(R[0, 2], R[2, 2]))

    # Goal camera centre in our frame is -R^T t (direction only: monocular, no scale).
    c = (-R.T @ t).ravel()
    c = c / (np.linalg.norm(c) + 1e-12)
    bearing_right = math.degrees(math.atan2(c[0], c[2]))

    keep = mask.ravel() > 0
    if keep.sum() < 8:
        return None
    H = K @ R @ np.linalg.inv(K)
    moved = cv2.perspectiveTransform(p_cur[keep].reshape(-1, 1, 2), H).reshape(-1, 2)
    parallax = float(np.median(np.linalg.norm(moved - p_goal[keep], axis=1)))
    return yaw_left, bearing_right, float(c[2]), parallax, int(inliers)


class GoalMatcher:
    """SIFT features for the four goal views, matched against each live frame."""

    def __init__(self, K: np.ndarray, targets, params: Params) -> None:
        self.K = K
        self.p = params
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.goal = [self._features(t) for t in targets]
        self.counts: list[int] = [0] * len(self.goal)   # ratio-test survivors per view, last frame

    def _features(self, image: np.ndarray):
        kp, des = self.sift.detectAndCompute(image, None)
        if des is None or len(kp) < 2:
            return np.empty((0, 2), np.float32), None
        des = np.sqrt(des / np.maximum(des.sum(axis=1, keepdims=True), 1e-12))  # RootSIFT
        return np.float32([k.pt for k in kp]), des.astype(np.float32)

    def estimate(self, image: np.ndarray) -> Estimate | None:
        self.counts = [0] * len(self.goal)
        pts, des = self._features(image)
        if des is None:
            return None
        best: Estimate | None = None
        for view, (goal_pts, goal_des) in enumerate(self.goal):
            if goal_des is None:
                continue
            pairs = self.bf.knnMatch(des, goal_des, k=2)
            good = [m for pr in pairs if len(pr) == 2 for m, n in [pr] if m.distance < self.p.ratio * n.distance]
            self.counts[view] = len(good)
            if len(good) < self.p.min_matches:
                continue
            p_cur = pts[[m.queryIdx for m in good]]
            p_goal = goal_pts[[m.trainIdx for m in good]]
            try:
                pose = pose_from_points(p_cur, p_goal, self.K, self.p.ransac_px, self.p.parallax_tol)
            except cv2.error:
                continue
            if pose is None:
                continue
            yaw_left, bearing, forward, parallax, inliers = pose
            if best is None or inliers > best.inliers:
                best = Estimate(
                    view=view,
                    matches=len(good),
                    inliers=inliers,
                    yaw_left=yaw_left,
                    yaw_front=wrap(yaw_left - VIEW_HEADING[view]),
                    bearing_right=bearing,
                    forward=forward,
                    parallax=parallax,
                )
        return best


class FinalApproach:
    """``step(image, steps_left)`` -> a move, ``Action.CHECKIN``, or None (not enough to go on)."""

    def __init__(self, K: np.ndarray, targets, params: Params | None = None) -> None:
        self.p = params or Params()
        self.K = np.asarray(K, dtype=np.float64)
        self.matcher = GoalMatcher(self.K, targets, self.p)
        self.rows: list[dict] = []
        self.status = "idle"
        self.last: Estimate | None = None

    def step(self, image: np.ndarray, steps_left: int | None = None):
        est = self.matcher.estimate(image)
        self.last = est
        if est is None or est.inliers < self.p.min_inliers:
            self.status = f"no lock, matches per view {self.matcher.counts}"
            self._log(est, None, self.status)
            return None
        if steps_left is not None and steps_left <= self.p.panic_steps:
            move, note = Action.CHECKIN, "out of steps"
        else:
            move, note = self.decide(est)
        self.status = f"{note}  (view {est.view}, {est.inliers} inl, par {est.parallax:.1f}px)  {self.matcher.counts}"
        self._log(est, move, note)
        return move

    def decide(self, est: Estimate):
        p = self.p
        if est.parallax > p.parallax_tol:
            if math.isnan(est.bearing_right):
                return None, "off the goal but no direction"
            # Not on the goal position yet: head for it. Forward if it is ahead, otherwise reverse.
            b = est.bearing_right
            forward = abs(b) <= 90.0
            err = b - (0.0 if forward else math.copysign(180.0, b))
            if abs(err) > p.bearing_tol:
                return self._turn(-err), f"turn toward goal ({b:+.0f} deg)"
            move = Action.FORWARD if forward else Action.BACKWARD
            return (move, p.drive_burst), "drive in" if forward else "back up"

        # On the position (to within parallax_tol): fix the heading against view 0.
        e = est.yaw_front
        if abs(e) > p.yaw_tol:
            return self._turn(e), f"align heading ({e:+.0f} deg)"
        if est.inliers < p.checkin_inliers:
            return None, "aligned, match too weak to check in"
        return Action.CHECKIN, "check in"

    def _turn(self, deg_left: float):
        action = Action.LEFT if deg_left > 0 else Action.RIGHT
        n = 1
        if self.p.turn_deg_per_tick:
            # aim to undershoot: the estimate is re-taken every request anyway
            n = int(np.clip(round(0.7 * abs(deg_left) / self.p.turn_deg_per_tick), 1, 4))
        return (action, n)

    def _log(self, est: Estimate | None, move, note: str) -> None:
        row = {"i": len(self.rows), "note": note, "move": str(move)}
        if est is not None:
            row.update(asdict(est))
        for i, c in enumerate(self.matcher.counts):
            row[f"m{i}"] = c
        self.rows.append(row)

    def dump_csv(self, path: Path) -> None:
        if not self.rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in self.rows for k in r}, key=lambda k: (k not in ("i", "note", "move"), k))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.rows)


def intrinsics(camera, image_shape) -> np.ndarray:
    """Best effort 3x3 K from ``info.camera`` (field names are not documented in what I have)."""

    def get(name):
        if isinstance(camera, dict):
            return camera.get(name)
        return getattr(camera, name, None)

    for name in ("K", "matrix", "intrinsics", "intrinsic_matrix", "camera_matrix", "k"):
        value = get(name)
        if value is not None:
            K = np.asarray(value, dtype=np.float64)
            if K.shape == (3, 3):
                return K
    if all(get(n) is not None for n in ("fx", "fy", "cx", "cy")):
        return np.array([[get("fx"), 0, get("cx")], [0, get("fy"), get("cy")], [0, 0, 1]], float)
    try:
        K = np.asarray(camera, dtype=np.float64)
        if K.shape == (3, 3):
            return K
    except (TypeError, ValueError):
        pass
    h, w = image_shape[:2]
    f = (w / 2) / math.tan(math.radians(45))
    print(f"WARNING: could not read intrinsics from info.camera = {camera!r}; assuming 90 deg FOV")
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
