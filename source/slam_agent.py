"""
SLAM agent: route_agent.py + a SLAM map of the maze, so the route can use EVERY place where the drives cross.

    uv run python source/slam_agent.py --local 22

Why
---
route_agent only links two drives where its strict overlap check finds them (a few hundred pairs, many near the spawn).
Everywhere else the route has to follow a recorded drive -- with all of the explorer's detours -- until it reaches one of
those links. Most places where two drives pass through the same corridor were never linked.

What this adds (slam_map.py does the work, once per maze, cached)
----------------------------------------------------------------
1. A pose for every recorded frame by robust least squares over: the robot's motion model (learned from the local
   simulator's physics: slam_calibrate.py), the rotation between neighbouring frames measured from their pictures
   (planar-motion visual odometry, ~0.35 degrees median), route_agent's verified links, and the shared spawn.
2. Loop-closure search: frame pairs the map puts within SLAM_CLOSE_M of each other are checked by their pictures;
   the real ones become new links, the map is solved again, and that finds more.
3. Those new links go into route_agent's graph exactly like its own verified links, so the shortest path (Dijkstra,
   from every frame to the goal, computed once) can cross between drives anywhere they meet.

Measured offline on maze 22 against the simulator's truth (cache/slam_truth, never used by the agent): median route
from a frame to the goal 156 -> 95 frames, true walking distance 10.1 m -> 7.6 m; routes that would cross a wall
47 -> 54 of 419.

Everything else -- localisation, hints, goal vote, keys -- is route_agent's, unchanged.
First start on a maze: route_agent's own caches + about 4-6 minutes of SLAM. Later starts load cache/<maze>/slam_*.json.
A picture of the map is written to cache/<maze>/slam_map.png (white = route from the spawn to the goal, red = goal).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cli
import cv2
import numpy as np
from vis_nav_sdk import run

import route_agent as R
import slam_map as M
import tracking_agent as T
from baseline_agent import _distance
from route_agent import RouteAgent, _env

SLAM = _env("SLAM", 1)                          # 0 = behave exactly like route_agent (for comparison)
SLAM_CACHE_VERSION = 1                          # bump to force the SLAM map to be rebuilt
SLAM_MIN_INL = _env("SLAM_LINK_MIN_INL", 30)    # a SLAM closure becomes a route link only with this many keypoint matches
SLAM_LINK_EXTRA = _env("SLAM_LINK_EXTRA", 0.5)  # extra route cost of a SLAM link over a route_agent link (they are a little less certain)
COMPASS = _env("COMPASS", 1)                    # 1 = compass tile under the camera (points where the hint says to go)
HIGHLIGHT = _env("HIGHLIGHT", 1)                # 1 = copy of your live view with "head here" marked on it

_GREEN, _AMBER, _RED, _GREY = (90, 200, 110), (60, 180, 245), (80, 80, 230), (140, 140, 140)   # BGR
_TILE = {"ok": (90, 200, 110), "turn": (245, 180, 60), "lost": (230, 80, 80)}                  # RGB for labels


def _hint_state(text: str):
    """(kind, colour key) from the hint text the agent shows."""
    t = text or ""
    for k in ("LEFT", "RIGHT", "FORWARD", "BACKWARD"):
        if t.startswith(k):
            return k, ("turn" if k in ("LEFT", "RIGHT") else "ok")
    if t.startswith(("lost", "not sure", "blocked", "no known route")):
        return "LOST", "lost"
    return "GOAL", "ok"


def _arrow(img, centre, angle_deg, length, colour, thick=6):
    """Arrow from centre; angle 0 = straight up (ahead), + = to the right."""
    a = math.radians(angle_deg)
    tip = (int(centre[0] + length * math.sin(a)), int(centre[1] - length * math.cos(a)))
    cv2.arrowedLine(img, (int(centre[0]), int(centre[1])), tip, colour, thick, cv2.LINE_AA, tipLength=0.3)


class SlamAgent(RouteAgent):
    _slam_P = None
    _slam_D = None
    _live = None

    # ------------------------------------------------------------------ on-screen guidance
    def _localise(self, image: np.ndarray) -> None:
        self._live = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if T.LIVE_IMAGE_IS_RGB else image
        super()._localise(image)

    def _go_angle(self) -> tuple[str, str, float]:
        """(kind, colour key, degrees to turn: 0 = straight ahead, + = right) -- the same advice as the hint text."""
        kind, col = _hint_state(self._hint())
        deg = float(np.clip(self.turn_need * T.TURN_PX * T.DEG_PER_PX, -150, 150))
        if kind == "FORWARD":
            deg = float(np.clip(deg, -30, 30))
        elif kind == "BACKWARD":
            deg = 180.0
        elif kind in ("LEFT", "RIGHT"):
            deg = (1 if kind == "RIGHT" else -1) * max(abs(deg), 20.0)    # never draw a turn as "straight on"
        return kind, col, deg

    def _compass_tile(self):
        kind, col, deg = self._go_angle()
        img = np.full((240, 320, 3), 30, np.uint8)
        c, r = (160, 130), 95
        cv2.circle(img, c, r, (70, 70, 70), 2, cv2.LINE_AA)
        for k in range(12):                                        # ticks every 30 degrees, longer at front/side/back
            a = math.radians(k * 30)
            L = 14 if k % 3 == 0 else 7
            p1 = (int(c[0] + r * math.sin(a)), int(c[1] - r * math.cos(a)))
            p2 = (int(c[0] + (r - L) * math.sin(a)), int(c[1] - (r - L) * math.cos(a)))
            cv2.line(img, p1, p2, (110, 110, 110), 2, cv2.LINE_AA)
        cv2.putText(img, "ahead", (c[0] - 22, c[1] - r - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        colour = {"ok": _GREEN, "turn": _AMBER, "lost": _RED}[col]
        if kind == "LOST":
            cv2.putText(img, "?", (c[0] - 14, c[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 1.6, _RED, 3, cv2.LINE_AA)
            label = "lost -- look around"
        elif kind == "GOAL":
            cv2.circle(img, c, 26, _GREEN, -1, cv2.LINE_AA)
            label = "at the goal?"
        else:
            _arrow(img, c, deg, r - 18, colour)
            cv2.circle(img, c, 7, (220, 220, 220), -1, cv2.LINE_AA)
            label = self._hint()
        words = label if len(label) <= 34 else label[:33] + "."
        cv2.putText(img, words, (8, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        return img, f"compass · {kind.lower()}", _TILE[col]

    def _highlight_tile(self):
        """Your live view with the part a route frame further ahead also sees boxed ("head here")."""
        if self._live is None:
            return None
        img = self._live.copy()
        h, w = img.shape[:2]
        kind, col, deg = self._go_angle()
        steer = getattr(self, "_steer", None)
        label = "no route frame in view"
        if steer is not None and self._qf is not None and not self.lost:
            target = steer[2]
            try:
                A, _ = T._inlier_points(self._qf, self._node_feats(target))
            except Exception:
                A = np.zeros((0, 3))
            if len(A) >= 4:
                x0, x1 = np.percentile(A[:, 0], [10, 90])
                y0, y1 = np.percentile(A[:, 1], [10, 90])
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), _GREEN, 3, cv2.LINE_AA)
                for x, y in A[:, :2]:
                    cv2.circle(img, (int(x), int(y)), 2, _GREEN, -1, cv2.LINE_AA)
                ahead = self.path.index(target) if target in self.path else 0
                label = f"head here ({ahead} steps ahead)"
                if kind == "FORWARD":           # walking: point the arrow at the box
                    cx = (x0 + x1) / 2
                    deg = float(np.clip((cx - w / 2) / (w / 2) * 45, -45, 45))
        if kind not in ("LOST", "GOAL"):
            colour = {"ok": _GREEN, "turn": _AMBER, "lost": _RED}[col]
            _arrow(img, (w // 2, h - 12), deg if kind != "BACKWARD" else 180, min(h, w) * 0.28, colour, 4)
        elif kind == "LOST":
            label = "lost -- look around"
        return img, label, _TILE[col]

    def _render_panel(self):
        tiles = list(super()._render_panel())
        extra = []
        try:
            if COMPASS:
                extra.append(self._compass_tile())
            if HIGHLIGHT:
                t = self._highlight_tile()
                if t is not None:
                    extra.append(t)
        except Exception as exc:          # guidance tiles must never break the run
            print("guidance tiles skipped:", exc)
        return extra + tiles

    def _build_graph(self, bounds, top_k):
        graph = super()._build_graph(bounds, top_k)
        if not SLAM:
            print("SLAM=0: route_agent graph only")
            return graph
        n = len(self.database)
        links = {(min(a, b), max(a, b)): (0, float(d.get("dx", 0.0)) if a < b else -float(d.get("dx", 0.0)))
                 for a, b, d in graph.edges(data=True)
                 if d.get("visual") and not d.get("same") and not d.get("ext")}
        cache = self._cache_dir / f"slam_v{SLAM_CACHE_VERSION}{R.CACHE_TAG}_{n}_{len(links)}.json"
        found, poses = None, None
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                found, poses = [tuple(x) for x in data["closures"]], np.array(data["poses"], float)
                print(f"slam: loaded the map and {len(found)} loop closures from {cache.name}")
            except Exception:
                found = None
        if found is None:
            t0 = time.time()
            print("slam: building the map of the maze (first run on this maze: a few minutes) ...", flush=True)
            D, P, Q, pairs = M.build(self.frames, self._node_feats, links, width=self._width)
            self._save_store()
            found = []
            for a, b in pairs:
                A, B = T._inlier_points(self._node_feats(a), self._node_feats(b))
                inl, dx, ls = T._scale_stats(A, B)
                found.append((int(a), int(b), int(inl), float(dx), float(ls)))
            poses = Q
            try:
                cache.write_text(json.dumps({"closures": found, "poses": np.round(Q, 4).tolist()}))
            except Exception as exc:
                print("slam: could not cache the map:", exc)
            print(f"slam: map built in {time.time() - t0:.0f}s")
        self._slam_P = poses
        added = skipped = 0
        for a, b, inl, dx, ls in found:
            if inl < SLAM_MIN_INL or abs(ls) > R.LINK_MAX_DEPTH or graph.has_edge(a, b):
                skipped += 1
                continue
            w = R.LINK_WEIGHT + SLAM_LINK_EXTRA + 0.5 * _distance(float(self.database[a] @ self.database[b])) + R.LINK_DEPTH_COST * abs(ls)
            graph.add_edge(a, b, weight=w, visual=True, dx=dx, slam=True)
            graph.add_edge(b, a, weight=w, visual=True, dx=-dx, slam=True)
            added += 1
        print(f"slam: {added} new links between places the drives share added to the route graph "
              f"({skipped} skipped: too few matches, too different in distance, or already linked)")
        return graph

    def _report_route(self) -> None:
        super()._report_route()
        if self._slam_P is None:
            return
        keep = {k: getattr(self, k) for k in ("_rs_node", "_rs_g", "_rs_path", "_route", "_route_goal", "_route_pos",
                                               "_jump_tick", "_last_jump")}
        try:
            route = list(self._route_from(self._starts[0])) if self._dist is not None else []
            for k, v in keep.items():          # drawing the map must not change what the route follower remembers
                setattr(self, k, v)
            Q = self._slam_P
            out = self._cache_dir / "slam_map.png"
            M.draw(Q, [f.trajectory for f in self.frames], out,
                   route=Q[route] if len(route) > 1 else None, goal=Q[self.goal] if 0 <= self.goal < len(Q) else None)
            jumps = sum(1 for a, b in zip(route[:-1], route[1:]) if self.graph[a][b].get("slam"))
            print(f"slam: route from the spawn to the goal: {max(0, len(route) - 1)} steps ({jumps} through SLAM links); "
                  f"map picture: {out}")
        except Exception as exc:
            print("slam: could not draw the map:", exc)


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", help="exploration data directory (default: download)")
    parser.add_argument("--subsample", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size")
    parser.add_argument("--top-k", type=int, default=100, help="unused (kept for the same command line)")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = SlamAgent(
        cli.exploration_data(args, args.data),
        n_clusters=args.n_clusters,
        subsample=args.subsample,
        top_k_shortcuts=args.top_k,
        cache_dir=Path("cache") / challenge.replace(":", "-"),
    )
    run(agent, challenge, **cli.run_options(args))
