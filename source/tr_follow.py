"""
Live route follower (teach and repeat). Knows ONE route (a list of exploration frames from the spawn to the goal)
and where along it you are. Each new camera frame is matched only against route frames near your current place,
plus a sparse look further ahead (so cutting a corner or skipping a loop the explorer made is picked up), never
against the rest of the maze -- look-alike places elsewhere cannot make it jump.

Hints come from geometry the images measure well: the turn between your view and the route frame you are at,
and the route's own heading changes ahead (measured between consecutive recorded frames).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from tr_map import DEG_PER_TICK, TRMap, ratio_pairs, rootsift, rotation_fit

NEAR_BACK = 6            # route frames behind the current place that are checked
NEAR_AHEAD = 30          # ... and ahead (all of them, via the thumbnail pre-filter)
FAR_AHEAD = 100000       # sparse look-ahead over the rest of the route (maze 77: a real skip of 319 frames)
FAR_STEP = 3
VERIFY = 8               # SIFT-verify this many best thumbnail candidates per frame
ACCEPT = 15              # inliers to accept a nearby route frame
ACCEPT_FAR = 30          # inliers to jump ahead beyond NEAR_AHEAD ...
FAR_CONFIRM = 2          # ... seen this many frames in a row
FAR_PARALLAX = 25.0      # px: a far jump must be (nearly) the same position, not just shared features
LOST_AFTER = 4           # frames without an accepted match before saying "lost"
TURN_SAY = 30.0          # deg: below this the view counts as lined up (replay of a real drive: 20 gave twice the false turns)
LOOK_M = 0.25            # steer toward the route point this far ahead (m); 0.5 cut corners in 0.2 m corridors
TURN_EVENT = 50.0        # deg: a change of travel direction this large ahead is announced as "then turn"
M_PER_TICK = 0.029
HINT_HOLD = 2            # a new kind of hint must repeat this many frames before it is shown
TURN_MEMORY_TICKS = 10   # after losing the route, turn guidance stays valid until you drive this many ticks
THUMB = (32, 24)
THUMB_SHIFT = 6


def _thumb(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def _ncc_shift(live: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Best normalised correlation of one thumbnail against a bank, over horizontal shifts."""
    W = THUMB[0]
    best = np.full(len(bank), -1.0, np.float32)
    for s in range(-THUMB_SHIFT, THUMB_SHIFT + 1):
        a = live[:, max(0, s):W + min(0, s)].ravel()
        b = bank[:, :, max(0, -s):W + min(0, -s)].reshape(len(bank), -1)
        a = (a - a.mean()) / (np.linalg.norm(a - a.mean()) + 1e-6)
        b = b - b.mean(1, keepdims=True)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
        np.maximum(best, b @ a, out=best)
    return best


@dataclass
class Fix:
    i: int              # route index
    inliers: int
    yaw: float          # deg, + = the route frame looks further LEFT than you (turn left to line up)
    parallax: float


class Follower:
    def __init__(self, m: TRMap, route: list[int], K: np.ndarray) -> None:
        self.m, self.route, self.K = m, route, K
        n = len(route)
        # heading along the route (deg, left +) and ticks from the start
        self.H = np.zeros(n)
        self.T = np.zeros(n)
        self.fwd = np.zeros(n)                 # forward ticks from the start (estimate, from the labels)
        links = {}
        for a, b, yaw, *_ in m.links:
            links[(a, b)] = yaw
            links[(b, a)] = -yaw
        for k in range(1, n):
            u, v = route[k - 1], route[k]
            ru, rv = m.records[u], m.records[v]
            if rv.drive == ru.drive and v == u + 1:
                dy = m.seq_yaw[u] or 0.0
                f = rv.ticks if "FORWARD" in rv.action else 0
                t = rv.ticks
            elif rv.drive == ru.drive and v == u - 1:
                dy = -(m.seq_yaw[v] or 0.0)
                f, t = 0, ru.ticks
            else:
                dy = links.get((u, v), 0.0)
                f, t = 0, abs(dy) / DEG_PER_TICK
            self.H[k] = self.H[k - 1] + dy
            self.T[k] = self.T[k - 1] + t
            self.fwd[k] = self.fwd[k - 1] + f
        # local path: heading from the images, distance from the forward records (good over short stretches)
        self.P = np.zeros((n, 2))
        for k in range(1, n):
            step = (self.fwd[k] - self.fwd[k - 1]) * M_PER_TICK
            h = math.radians(self.H[k])
            self.P[k] = self.P[k - 1] + step * np.array([math.cos(h), math.sin(h)])
        self.S = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.P, axis=0), axis=1))])  # path length
        self.B = np.array([self._bearing(k) for k in range(n)])      # travel direction at each route index
        self.thumbs = np.stack([_thumb(cv2.imread(m.records[g].path, cv2.IMREAD_GRAYSCALE)) for g in route])
        self.p = 0
        self.fix: Fix | None = None
        self.misses = 0
        self.far_pending: tuple[int, int] | None = None
        self._shown = ("start", "Drive along the route: follow the hints")
        self._cand = None
        self._cand_n = 0
        self.prev_live = None          # features of the previous camera frame (for the turn between frames)
        self.turned = 0.0              # deg turned LEFT since the last fix
        self.moved = 0.0               # forward/backward ticks since the last fix
        self.last_need = None          # turn that was needed at the last fix (deg, left +)

    # ------------------------------------------------------------------ localisation
    def _candidates(self, live_thumb: np.ndarray, wide: bool) -> list[int]:
        n = len(self.route)
        lo, hi = max(0, self.p - NEAR_BACK), min(n, self.p + NEAR_AHEAD + 1)
        idx = list(range(lo, hi))
        far_hi = n if wide else min(n, self.p + FAR_AHEAD)
        idx += list(range(hi, far_hi, FAR_STEP))
        if wide:
            idx += list(range(0, lo, FAR_STEP))
        idx = np.array(sorted(set(idx)))
        score = _ncc_shift(live_thumb, self.thumbs[idx])
        top = idx[np.argsort(-score)[:VERIFY]]
        # always verify the next few route frames too: the thumbnail can miss a turned view
        near = [i for i in range(self.p, min(n, self.p + 4))]
        return list(dict.fromkeys(list(near) + [int(i) for i in top]))

    def localise(self, image: np.ndarray, moved_ticks: float = 0.0, key_turn: float | None = None) -> Fix | None:
        """moved_ticks: forward/backward ticks sent since the previous frame; key_turn: deg LEFT the keys asked for
        (fallback when the images can't measure the turn)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        live = rootsift(gray)
        if self.prev_live is not None:
            a, b = ratio_pairs(self.prev_live, live)
            r = rotation_fit(a, b, self.K) if a is not None else None
            if r is not None and r[2] >= 15:
                self.turned += r[0]
            elif key_turn is not None:
                self.turned += key_turn
        self.prev_live = live
        self.moved += abs(moved_ticks)
        wide = self.misses >= LOST_AFTER
        best: Fix | None = None
        for i in self._candidates(_thumb(gray), wide):
            a, b = ratio_pairs(live, self.m.feats[self.route[i]])
            r = rotation_fit(a, b, self.K) if a is not None else None
            if r is None:
                continue
            yaw, par, inl, _ = r
            far = i > self.p + NEAR_AHEAD or i < self.p - NEAR_BACK
            if inl < (ACCEPT_FAR if far else ACCEPT):
                continue
            if far and par > FAR_PARALLAX:          # different position that shares features (maze 42: 69 px)
                continue
            # prefer ahead over behind at similar strength (you came from behind); when lost, stay near the last
            # sure place unless a far frame is clearly better (look-alikes along the route itself exist)
            score = inl + (2 if i >= self.p else 0) - (0.15 if wide else 0.05) * abs(i - self.p)
            if best is None or score > best[0]:
                best = (score, Fix(i, inl, yaw, par))
        if best is None:
            self.misses += 1
            self.fix = None
            return None
        fix = best[1]
        far = fix.i > self.p + NEAR_AHEAD or fix.i < self.p - NEAR_BACK
        if far:                                          # a far jump is always confirmed, lost or not
            n_seen = self.far_pending[1] + 1 if self.far_pending and abs(self.far_pending[0] - fix.i) <= 6 else 1
            self.far_pending = (fix.i, n_seen)
            if n_seen < FAR_CONFIRM:
                self.misses += 1
                self.fix = None
                return None
        self.far_pending = None
        self.misses = 0
        self.p = fix.i
        self.fix = fix
        self.last_need = fix.yaw + (self.B[fix.i] - self.H[fix.i])
        self.turned = 0.0
        self.moved = 0.0
        return fix

    # ------------------------------------------------------------------ hints
    def _ahead(self, i: int, dist: float) -> int:
        k = int(np.searchsorted(self.S, self.S[i] + dist))
        return min(len(self.route) - 1, k)

    def _bearing(self, i: int) -> float:
        """Direction of travel from route index i (deg, same frame as H): toward the point LOOK_M further along."""
        k = self._ahead(i, LOOK_M)
        d = self.P[k] - self.P[i]
        if np.linalg.norm(d) < 0.05:
            return float(self.H[k])
        b = math.degrees(math.atan2(d[1], d[0]))
        return float(self.H[i] + ((b - self.H[i] + 180) % 360 - 180))   # unwrap near H[i]

    def remaining_records(self) -> int:
        return len(self.route) - 1 - self.p

    def remaining_ticks(self) -> float:
        return float(self.T[-1] - self.T[self.p])

    def next_turn(self, i: int) -> int | None:
        """Route index where the travel direction has changed by TURN_EVENT from here, or None."""
        for t in range(i + 1, len(self.route)):
            if abs(self.B[t] - self.B[i]) >= TURN_EVENT:
                return t
        return None

    def turn_after(self, t: int, i: int) -> float:
        """Size of the turn at route index t: travel direction after it settles minus the direction at i."""
        k = self._ahead(t, 0.3)
        return float(self.B[k] - self.B[i])

    def hint(self) -> tuple[str, str]:
        """(kind, text). kind is used for the hold-before-change rule."""
        if self.fix is None:
            if self.misses >= LOST_AFTER and self.remaining_records() <= 25:
                return "lost", "NEAR THE GOAL: look for the goal view (top right); when you see it, let go and wait for READY"
            if self.misses >= LOST_AFTER and self.last_need is not None and self.moved <= TURN_MEMORY_TICKS:
                need = self.last_need - self.turned
                if abs(need) >= TURN_SAY * 0.5:
                    side = "LEFT" if need > 0 else "RIGHT"
                    return f"re{side}", (f"OFF THE ROUTE: TURN {side} ~{round(abs(need) / DEG_PER_TICK)} presses "
                                         f"({abs(need):.0f} deg) to face it again")
                return "refwd", "FACING THE ROUTE AGAIN: FORWARD slowly until YOU ARE HERE matches"
            if self.misses >= LOST_AFTER:
                return "lost", "NOT SURE WHERE YOU ARE: turn slowly on the spot until YOU ARE HERE matches"
            return self._shown
        i = self.fix.i
        if self.remaining_records() <= 3:
            return "goal", "GOAL AHEAD: let go of the keys and wait for READY"
        need = self.fix.yaw + (self.B[i] - self.H[i])     # + = turn left: line up with the route's direction
        if abs(need) >= 150:
            return "around", f"TURN AROUND ~{round(abs(need) / DEG_PER_TICK)} presses ({abs(need):.0f} deg)"
        if abs(need) >= TURN_SAY:
            side = "LEFT" if need > 0 else "RIGHT"
            return f"turn{side}", f"TURN {side} ~{round(abs(need) / DEG_PER_TICK)} presses ({abs(need):.0f} deg)"
        t = self.next_turn(i)
        if t is None:
            ahead = (self.S[-1] - self.S[i]) / M_PER_TICK
            return "fwd", f"FORWARD (~{ahead:.0f} presses to the goal area)"
        deg = self.turn_after(t, i)
        side = "LEFT" if deg > 0 else "RIGHT"
        presses = (self.S[t] - self.S[i]) / M_PER_TICK
        if abs(deg) >= 150:
            return "dead", (f"DEAD END AHEAD (~{presses:.0f} presses): the explorer turned around there. "
                            f"Skip it if you see the way on")
        if presses < 3:
            return f"soon{side}", f"TURN {side} NOW ~{round(abs(deg) / DEG_PER_TICK)} presses ({abs(deg):.0f} deg)"
        return f"fwd{side}", f"FORWARD ~{presses:.0f} presses, then turn {side} ({abs(deg):.0f} deg)"

    def stable_hint(self) -> str:
        kind, text = self.hint()
        if kind == self._shown[0] or kind in ("lost", "goal"):
            self._shown = (kind, text)
            self._cand, self._cand_n = None, 0
        elif self._cand == kind:
            self._cand_n += 1
            if self._cand_n >= HINT_HOLD - 1:
                self._shown = (kind, text)
                self._cand, self._cand_n = None, 0
        else:
            self._cand, self._cand_n = kind, 0
        return self._shown[1]
