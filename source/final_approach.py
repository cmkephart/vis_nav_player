"""
Phase B: the last metre.

Place recognition (the baseline's VLAD) says which exploration frame we are near. This module
takes over from there. It matches the live frame against the four goal photos, recovers the
relative camera pose with an essential matrix, and turns that into moves:

    off the spot -> turn toward the goal's bearing, then drive in
    on the spot (a strong match on any view) -> turn until view 0 is dead ahead, check in

The goal photos only match within ~0.3 m of the goal (calibrate.py, maze 7): whatever drives
the robot there has to bring it that close.

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
    """Thresholds from calibrate.py on maze 7 (1666 frames with the true pose), see the notes."""
    ratio: float = 0.8              # Lowe ratio test
    min_matches: int = 12           # ratio-test survivors needed to try a pose
    min_inliers: int = 15           # below this the estimate is ignored (module returns None)
    engage_front: int = 15          # front view >= 15: never seen beyond 0.22 m from the goal
    # "on the spot" per view (front, left, back, right): the 95th-percentile distance is ~0.10-0.13 m.
    # The left view matches strongly from much further (35 inliers at 0.25 m), hence its high bar.
    spot_inliers: tuple = (35, 80, 50, 35)
    checkin_inliers: int = 35       # front >= 35 with |yaw| <= 3: worst 0.082 m over 3 runs (full points <= 0.1)
    checkin_fallback: int = 30      # after max_nudges, accept front >= 30: 3 of 112 frames were 0.10-0.105 m
    scan_ticks: int = 5             # no lock: turn this much (~17 deg) and look again ...
    max_scans: int = 22             # ... for about a full circle, then give up (return None)
    ransac_px: float = 1.0
    parallax_tol: float = 4.0       # px; a pure rotation this good = no direction to give (bearing NaN)
    yaw_tol: float = 3.0            # deg; yaw_front is off by ~0.5 deg (median) near the goal
    bearing_tol: float = 20.0       # deg; the bearing is good to ~3 deg between 0.05 and 0.3 m
    drive_burst: int = 1            # ticks per FORWARD/BACKWARD request (~0.028 m per tick)
    turn_deg_per_tick: float = 3.44 # measured: 3.44 +- 0.06 deg per LEFT/RIGHT tick
    max_turn_ticks: int = 30        # one request turns at most this much, then re-measure
    max_legs: int = 3               # straight legs toward the goal before accepting checkin_fallback;
                                    # at twice this, check in anyway
    max_leg_ticks: int = 10         # one leg drives at most this far (~0.29 m) before re-checking
    nudge_axis_tol: float = 45.0    # drive straight forward/back (no turning) if the goal is within this of the axis
    probe_ticks: int = 3            # first lock: turn this much and check the estimate follows ...
    confirm_tol: float = 10.0       # ... to within this many deg (real goal: 90-96% pass; a lookalike
                                    # 1.56 m away jumped 40-190 deg). Two misses = lookalike.
    probe_attempts: int = 3         # misses before calling it a lookalike (2 rejected a real match at 0.16 m)
    reject_frames: int = 30         # after rejecting a lookalike, ignore matches for this many frames
    tick_budget: int = 120          # phase B stops and hands control back after spending this many ticks
                                    # (SPL counts every tick; a stuck phase B once burned 1032)
    panic_steps: int = 3            # check in anyway when this few steps are left


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


def load_goal_views(data_dir: Path) -> list[np.ndarray]:
    """Goal photos from a dataset folder (target*.jpg/png); a 4-view strip is split into views."""
    views: list[np.ndarray] = []
    for p in sorted(Path(data_dir).glob("target*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        n = max(1, round(w / h / (4 / 3)))  # 1280x240 = four 320x240 views: front, left, back, right
        tw = w // n
        views += [img[:, i * tw:(i + 1) * tw] for i in range(n)]
    return views


def place_scores(counts: np.ndarray, traj: list[int], window: int = 4, floor: int = 10) -> np.ndarray:
    """Per node: how many of the goal's views can be seen from the stretch of path around it.

    ``counts`` is (nodes, views) raw match counts. Each view contributes its best count within
    +-``window`` nodes of the same drive, minus a noise ``floor``, so one strong match on a
    single view scores less than modest matches on all four."""
    c = np.maximum(np.asarray(counts, dtype=float) - floor, 0.0)
    scores = np.zeros(len(c))
    for i in range(len(c)):
        idx = [j for j in range(max(0, i - window), min(len(c), i + window + 1)) if traj[j] == traj[i]]
        scores[i] = c[idx].max(axis=0).sum()
    return scores


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

    def raw_counts(self, image: np.ndarray) -> list[int]:
        """Ratio-test survivors against each goal view; no geometry, so it is fast."""
        _, des = self._features(image)
        out = []
        for _, goal_des in self.goal:
            if des is None or goal_des is None:
                out.append(0)
                continue
            pairs = self.bf.knnMatch(des, goal_des, k=2)
            out.append(sum(1 for pr in pairs if len(pr) == 2 and pr[0].distance < self.p.ratio * pr[1].distance))
        return out

    def estimate(self, image: np.ndarray) -> Estimate | None:
        self.counts = [0] * len(self.goal)
        self.per_view: dict[int, Estimate] = {}  # every view's estimate this frame, not just the best
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
            here = Estimate(
                view=view,
                matches=len(good),
                inliers=inliers,
                yaw_left=yaw_left,
                yaw_front=wrap(yaw_left - VIEW_HEADING[view]),
                bearing_right=bearing,
                forward=forward,
                parallax=parallax,
            )
            self.per_view[view] = here
            if best is None or inliers > best.inliers:
                best = here
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
        self.mode: str | None = None   # "align" or "move"; chosen afresh for each new lock
        self.leg: Action | None = None
        self.leg_ticks = 0
        self.legs = 0
        self.scans = 0
        self.confirmed = False  # the lock passed the probe-turn test
        self._expect: float | None = None  # yaw_front expected after the probe turn
        self._expect_view = 0
        self._misses = 0
        self._cooldown = 0
        self.spent = 0  # ticks executed since the last reset_budget(); the caller reports them via charge()

    def charge(self, move) -> None:
        """Called by whoever EXECUTES a move phase B returned, so only real ticks count."""
        if isinstance(move, tuple):
            self.spent += int(move[1])

    def reset_budget(self) -> None:
        """A fresh start: call when the driver takes over (a key press) or a new test begins."""
        self.spent = 0
        self.mode, self.leg, self.legs, self.scans = None, None, 0, 0
        self.confirmed, self._expect, self._misses = False, None, 0

    def step(self, image: np.ndarray, steps_left: int | None = None):
        est = self.matcher.estimate(image)
        self.last = est
        if self.spent >= self.p.tick_budget:
            self.status = f"TICK BUDGET USED ({self.spent}): your turn -- press any key"
            self._log(est, None, "tick budget used up")
            return None
        if self._cooldown > 0:
            self._cooldown -= 1
            self.status = f"ignoring matches after a lookalike ({self._cooldown} frames)"
            self._log(est, None, self.status)
            return None
        if est is None or est.inliers < self.p.min_inliers:
            self._expect = None  # a probe in flight is void
            # no lock: near the goal some view always matches from SOME heading, so look around
            self.scans += 1
            move = (Action.LEFT, self.p.scan_ticks) if self.scans <= self.p.max_scans else None
            if move is None:
                self.confirmed = False  # lost it for good: the next lock must prove itself again
            note = f"no lock, scan {self.scans}/{self.p.max_scans}" if move else "no lock, gave up"
            self.status = f"{note}, matches per view {self.matcher.counts}"
            self._log(est, move, note)
            return move
        self.scans = 0
        if steps_left is not None and steps_left <= self.p.panic_steps:
            move, note = Action.CHECKIN, "out of steps"
        elif not self.confirmed:
            move, note = self.confirm(est)
        else:
            move, note = self.decide(est)
        self.status = f"{note}  (view {est.view}, {est.inliers} inl, par {est.parallax:.1f}px)  {self.matcher.counts}"
        self._log(est, move, note)
        return move

    def confirm(self, est: Estimate):
        """Turn a known amount and check yaw_front moves with it. A real goal match does (to a few
        deg); a lookalike's pose is nonsense and jumps. Costs one small turn per new lock."""
        p = self.p
        if self._expect is not None:
            # judge the probe on the SAME goal view as before: the best view can switch between
            # frames, and a switch to the wrong view showed up as 82-169 deg "jumps" at the real goal
            same = self.matcher.per_view.get(self._expect_view)
            ref = same if same is not None and same.inliers >= p.min_inliers // 2 else est
            err = abs(wrap(ref.yaw_front - self._expect))
            self._expect = None
            if err <= p.confirm_tol:
                self.confirmed, self._misses = True, 0
                self.mode, self.leg, self.legs = None, None, 0
                return self.decide(est)
            self._misses += 1
            if self._misses >= self.p.probe_attempts:
                self._misses, self._cooldown = 0, p.reject_frames
                return None, f"lookalike: estimate jumped {err:.0f} deg after the probe turn"
        left = est.yaw_front >= 0  # probe toward alignment, so the turn isn't wasted
        self._expect = est.yaw_front - math.copysign(p.probe_ticks * p.turn_deg_per_tick, est.yaw_front or 1.0)
        self._expect_view = est.view
        return (Action.LEFT if left else Action.RIGHT, p.probe_ticks), "probe turn: is this the goal?"

    def sees_goal(self, est: Estimate | None) -> bool:
        """Close enough for phase B to take over: the front view at all, or any view strongly.
        (Other views at 15-34 inliers were seen up to 1.7 m away; the front view never beyond 0.22 m.)"""
        if est is None or self._cooldown > 0:
            return False
        return (est.view == 0 and est.inliers >= self.p.engage_front) or est.inliers >= self.p.spot_inliers[est.view]

    def decide(self, est: Estimate):
        """Two modes. ALIGN: on the spot, turn to face view 0 and check in, or decide to move.
        MOVE: drive straight legs along our own axis (turning only when the goal is more than
        nudge_axis_tol off it), each leg ending when the goal comes abeam. Turning costs far more
        ticks than driving (3.44 deg vs 0.029 m per tick), so the legs avoid re-aiming every step."""
        p = self.p
        spot = est.inliers >= p.spot_inliers[est.view] or math.isnan(est.bearing_right)
        if self.mode is None:
            self.mode = "align" if spot else "move"
        for _ in range(3):  # move -> align -> move at most
            if self.mode == "align":
                e = est.yaw_front  # accurate to ~0.5 deg here, from whichever view matched
                if abs(e) > p.yaw_tol:
                    return self._turn(e), f"align heading ({e:+.0f} deg)"
                front = est.inliers if est.view == 0 else 0
                if front >= p.checkin_inliers:
                    return Action.CHECKIN, "check in"
                if self.legs >= p.max_legs and front >= p.checkin_fallback:
                    return Action.CHECKIN, f"check in (fallback after {self.legs} legs)"
                if self.legs >= 2 * p.max_legs:
                    return Action.CHECKIN, "check in (gave up refining)"
                if math.isnan(est.bearing_right):
                    if front >= p.checkin_fallback:
                        return Action.CHECKIN, "check in (aligned, no direction to refine)"
                    self.legs += 1
                    return (Action.BACKWARD, p.drive_burst), "aligned, weak, no direction: step back to see"
                self.mode, self.leg = "move", None  # aligned but weak: a few cm off
                continue

            b = est.bearing_right  # mode == "move"
            if math.isnan(b):
                self.mode = "align"
                continue
            if self.leg is not None:
                still_ahead = (abs(b) < 90.0) == (self.leg == Action.FORWARD)
                if still_ahead and self.leg_ticks < p.max_leg_ticks:
                    self.leg_ticks += p.drive_burst
                    return (self.leg, p.drive_burst), f"leg {self.leg.name.lower()} (goal {b:+.0f} deg)"
                self.legs += 1  # the goal came abeam: closest point on this line
                self.mode, self.leg = "align", None
                continue
            ahead = abs(b) <= p.nudge_axis_tol
            behind = abs(b) >= 180.0 - p.nudge_axis_tol
            if ahead or behind:
                self.leg, self.leg_ticks = (Action.FORWARD if ahead else Action.BACKWARD), p.drive_burst
                return (self.leg, p.drive_burst), f"start leg {self.leg.name.lower()} (goal {b:+.0f} deg)"
            err = b - (0.0 if abs(b) <= 90.0 else math.copysign(180.0, b))
            return self._turn(-err), f"turn toward goal ({b:+.0f} deg)"
        return None, "no decision"

    def _turn(self, deg_left: float):
        action = Action.LEFT if deg_left > 0 else Action.RIGHT
        n = int(np.clip(round(abs(deg_left) / self.p.turn_deg_per_tick), 1, self.p.max_turn_ticks))
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
