"""
[STABLE route agent: keyboard hints along a verified route; reached the goal on every maze it was tested on (SPL about 1 of 4 points).

Run from the repo root:
    uv run python source/stable/route_agent.py --local 21]

Route agent: the same camera-only tracker as tracking_agent.py, but the MAP is rebuilt so the route is real.

What was wrong before
---------------------
The maze reuses the same wall posters in many places. A single frame pair that "matches" is therefore often two
DIFFERENT places showing the same poster. Those false shortcuts made the goal look ~30 steps away from almost
anywhere, so the hints led you towards a place that does not connect to the goal.

What this file changes
----------------------
A visual link between two frames is only kept if it is part of an OVERLAP: when two drives really pass through the
same corridor, the frames next to the linked pair match each other as well, in a long run
(i-6..i+6 against j-6..j+6, in the same or the opposite direction). One matching poster gives a lone pair; a real
overlap gives a whole diagonal of matches. Only the diagonals become links (every verified pair in the run is
added, so the route can cross at any point of the overlap, not just at one frame).

Everything else (localisation, depth cue, press calibration, H key, hint wording) is inherited from
tracking_agent.py unchanged. Run it exactly like the tracking agent:

    uv run python source/route_agent.py ...

The first start builds and caches the verified links (about 2-4 minutes). Later starts load them from cache/.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parent.parent))   # cli.py, vlad.py, baseline_agent.py, keyboard_agent.py live one folder up


import hashlib
import json
import os
import time
from pathlib import Path

import cli
import cv2
import networkx as nx
import numpy as np
from vis_nav_sdk import run

import tracking_agent as T
from tracking_agent import TrackingAgent, _distance, _ransac, _ratio_pairs, _scale_stats, match_wide, rootsift

# ----------------------------------------------------------------------------- tunable values
# One dial for how many shortcuts between recordings are accepted (more shortcuts = shorter routes, a little more
# risk of a wrong one). Change it with  $env:LINK_LOOSENESS=3  in PowerShell before the run, or edit the default here.
#   1 = strict (default, the version that worked for you)   2 = more links   3 = most links
LINK_LOOSENESS = int(os.environ.get("LINK_LOOSENESS", "1"))
_LEVELS = {            # candidates/frame, first-pair matches, neighbour-pair matches, supporting pairs, panel strips
    1: (12, 25, 15, 4, 5),
    2: (20, 22, 12, 3, 5),
    3: (30, 20, 10, 3, 4),
}
CAND_PER_NODE, SEED_MIN_INLIERS, RUN_PAIR_MIN_INLIERS, RUN_MIN_SUPPORT, _STRIPS = _LEVELS.get(LINK_LOOSENESS, _LEVELS[2])
CAND_MIN_GAP = 30         # ...ignoring frames this close in the recording (they are trivially similar)
CAND_MIN_SIM = 0.0        # ...and look-alikes with a VLAD similarity below this
RUN_HALF = 6              # the frames within +-this many steps of both ends are checked as well
LINK_WEIGHT = 2.0         # route cost of crossing a verified link (a recorded step costs 1.0)
ROUTE_BACKWARD_COST = 8.0 # walking backwards along a recording is blind (frames face the other way): last resort
ROUTE_LINK_RETRIES = 12   # verified links are trustworthy: only drop one after this many failed arrivals
GOAL_RANSAC_TOP = 40      # goal search: per goal view, this many best-looking frames get the geometric + layout check
GOAL_CLUSTER = 3          # a real place makes the frames within +-this many steps match as well
GOAL_CAP = 60             # one frame counts at most this many matches towards a cluster
GOAL_MIN_CLUSTER = 40     # a cluster weaker than this is not believed over the VLAD guess
LINKS_VERSION = 3         # bump to force the cache to be rebuilt
# -----------------------------------------------------------------------------------------------


def build_verified_links(n, database, feats, traj, log=print):
    """Return {(i, j): (inliers, dx)} (i < j) of links that are part of a real overlap.

    n: number of frames; database: (n, d) unit VLAD vectors; feats(i) -> (pts, desc); traj[i]: drive of frame i.
    """
    t0 = time.time()
    memo: dict[tuple[int, int], tuple[int, float]] = {}

    def m(i: int, j: int, width: float) -> tuple[int, float]:
        key = (i, j) if i < j else (j, i)
        if key not in memo:
            r = match_wide(feats(key[0]), feats(key[1]), width)
            memo[key] = r
        inl, dx = memo[key]
        return (inl, dx) if i < j else (inl, -dx)

    sim = database @ database.T
    np.fill_diagonal(sim, -2)
    for i in range(n):
        sim[i, max(0, i - CAND_MIN_GAP) : i + CAND_MIN_GAP + 1] = -2
    seeds = set()
    for i in range(n):
        for j in np.argpartition(-sim[i], CAND_PER_NODE)[:CAND_PER_NODE]:
            if sim[i, j] > CAND_MIN_SIM:
                seeds.add((min(i, int(j)), max(i, int(j))))
    log(f"route links: {len(seeds)} candidate pairs ({time.time() - t0:.0f}s)")

    width = 320.0
    links: dict[tuple[int, int], tuple[int, float]] = {}
    ordered = sorted(seeds)
    covered: set[tuple[int, int]] = set()
    for k, (i, j) in enumerate(ordered):
        if k and k % 3000 == 0:
            log(f"  checked {k}/{len(ordered)} pairs, {len(links)} links so far ({time.time() - t0:.0f}s)")
        if (i, j) in covered:
            continue
        inl, dx = m(i, j, width)
        if inl < SEED_MIN_INLIERS:
            continue
        best: list[tuple[int, int, int, float]] = []
        for direction in (1, -1):               # same direction of travel, or the other drive going the opposite way
            ok: list[tuple[int, int, int, float]] = []
            for d in range(-RUN_HALF, RUN_HALF + 1):
                a, b = i + d, j + direction * d
                if not (0 <= a < n and 0 <= b < n) or a == b or traj[a] != traj[i] or traj[b] != traj[j]:
                    continue
                if d == 0:
                    ok.append((a, b, inl, dx))
                    continue
                inl2, dx2 = m(a, b, width)
                if inl2 >= RUN_PAIR_MIN_INLIERS:
                    ok.append((a, b, inl2, dx2))
            if len(ok) > len(best):
                best = ok
        if len(best) >= RUN_MIN_SUPPORT:
            for a, b, inl2, dx2 in best:
                key = (a, b) if a < b else (b, a)
                links[key] = (inl2, dx2 if a < b else -dx2)
                covered.add(key)
    log(f"route links: {len(links)} verified pairs kept ({time.time() - t0:.0f}s)")
    return links


class RouteAgent(TrackingAgent):
    def _build_graph(self, bounds, top_k: int) -> nx.DiGraph:
        self._starts = [int(s0) for s0, _ in bounds]
        n = len(self.database)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))

        # 1. recorded steps (walking a FORWARD step backwards is blind, so it is expensive)
        for start, end in bounds:
            for i in range(start, end - 1):
                act = self.frames[i].action
                fw, bw = T.TEMPORAL_WEIGHT, T.TEMPORAL_WEIGHT
                if act == "FORWARD":
                    bw = ROUTE_BACKWARD_COST
                elif act == "BACKWARD":
                    fw = ROUTE_BACKWARD_COST
                graph.add_edge(i, i + 1, weight=fw)
                graph.add_edge(i + 1, i, weight=bw)

        probe = cv2.imread(str(self.frames[0].path))
        self._width = float(probe.shape[1]) if probe is not None else 640.0

        # 2. "same pose again" links inside one drive (the explorer wiggled and came back to the same view)
        rot_p = np.zeros(n + 1, int)
        fwd_p = np.zeros(n + 1, int)
        for k, f in enumerate(self.frames):
            rot_p[k + 1] = rot_p[k] + (f.action == "RIGHT") - (f.action == "LEFT")
            fwd_p[k + 1] = fwd_p[k] + (f.action == "FORWARD") - (f.action == "BACKWARD")
        same = 0
        for start, end in bounds:
            for i in range(start, end):
                for j in range(i + 2, min(end, i + T.SAME_POSE_SPAN + 1)):
                    if rot_p[j] != rot_p[i] or fwd_p[j] != fwd_p[i]:
                        continue
                    inl, dx = T.match(self._node_feats(i), self._node_feats(j))
                    if inl >= T.SAME_POSE_MIN_INLIERS and abs(dx) <= T.SAME_POSE_MAX_SHIFT:
                        graph.add_edge(i, j, weight=T.SAME_POSE_WEIGHT, visual=True, same=True, dx=dx)
                        graph.add_edge(j, i, weight=T.SAME_POSE_WEIGHT, visual=True, same=True, dx=-dx)
                        same += 1

        # 3. verified overlaps between drives / different parts of a drive (cached)
        cache = self._cache_dir / (
            f"route_links_v{LINKS_VERSION}_{n}_{CAND_PER_NODE}_{SEED_MIN_INLIERS}_{RUN_HALF}_{RUN_PAIR_MIN_INLIERS}_{RUN_MIN_SUPPORT}_{_STRIPS}.json"
        )
        links = None
        if cache.exists():
            try:
                links = {(a, b): (inl, dx) for a, b, inl, dx in json.loads(cache.read_text())}
                print(f"route links: loaded {len(links)} verified pairs from {cache.name}")
            except Exception:
                links = None
        if links is None:
            traj = [f.trajectory for f in self.frames]
            links = build_verified_links(n, self.database, self._node_feats, traj)
            self._save_store()
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps([[int(a), int(b), int(i), float(d)] for (a, b), (i, d) in links.items()]))
            except Exception as exc:
                print("could not cache links:", exc)
        for (a, b), (inl, dx) in links.items():
            w = LINK_WEIGHT + 0.5 * _distance(float(self.database[a] @ self.database[b]))
            graph.add_edge(a, b, weight=w, visual=True, dx=dx)
            graph.add_edge(b, a, weight=w, visual=True, dx=-dx)
        print(f"graph: {n} nodes, {graph.number_of_edges()} directed edges ({same} same-pose links, "
              f"{len(links)} verified overlap pairs)")
        return graph

    # ------------------------------------------------------------------ goal by voting
    def _goal_by_voting(self, targets) -> tuple[int, list]:
        """The VLAD guess for the goal view is nearly random (similarity ~0.1). Instead every frame is SIFT-matched
        against all four goal views (with the panel-layout check), and the goal is where a whole run of neighbouring
        frames matches: a real place gives a cluster, a look-alike poster gives one lonely frame."""
        n = len(self.frames)
        drive = np.array([f.trajectory for f in self.frames])
        votes = np.zeros((len(targets), n), np.float32)
        t0 = time.time()
        for v, img in enumerate(targets):
            qf = rootsift(img)
            pre = []
            for j in range(n):
                a, b = _ratio_pairs(qf, self._node_feats(j))
                pre.append((len(a), j, a, b))
            for cnt, j, a, b in sorted(pre, key=lambda t: -t[0])[:GOAL_RANSAC_TOP]:
                if cnt >= 12:
                    inl, _, _ = _scale_stats(*_ransac(a, b), float(img.shape[1]))
                    votes[v, j] = min(inl, GOAL_CAP)
        cluster = np.zeros(n, np.float32)
        for m in range(-GOAL_CLUSTER, GOAL_CLUSTER + 1):
            shifted = np.zeros_like(votes)
            same = np.zeros(n, bool)
            if m >= 0:
                shifted[:, : n - m] = votes[:, m:]
                same[: n - m] = drive[: n - m] == drive[m:]
            else:
                shifted[:, -m:] = votes[:, : n + m]
                same[-m:] = drive[-m:] == drive[: n + m]
            cluster += (shifted * same).sum(0)
        try:                                   # tie-break only: prefer the cluster whose look is also closest overall (VLAD)
            sim = np.max([self.database @ self.extractor.extract(t) for t in targets], axis=0)
            cluster = cluster + 15.0 * np.clip(sim, 0.0, 1.0)
        except Exception:
            pass
        order = np.argsort(-cluster)
        best = int(order[0])
        lo, hi = max(0, best - GOAL_CLUSTER), min(n, best + GOAL_CLUSTER + 1)
        centre = lo + int(np.argmax(votes[:, lo:hi].sum(0)))
        top: list[tuple[int, float]] = []
        for j in order:
            if all(abs(int(j) - k) > 20 for k, _ in top):
                top.append((int(j), float(cluster[j])))
            if len(top) == 3:
                break
        print(f"goal vote: {time.time() - t0:.0f}s, best clusters (node, score): {top}")
        return (centre if cluster[best] >= GOAL_MIN_CLUSTER else -1), top

    def _report_route(self) -> None:
        try:
            starts = getattr(self, "_starts", [0])
            full = min(nx.shortest_path_length(self.graph, s0, self.goal, weight="weight") for s0 in starts)
            plain = nx.subgraph_view(self.graph, filter_edge=lambda u, v: not self.graph[u][v].get("visual", False))
            rec = min(nx.shortest_path_length(plain, s0, self.goal, weight="weight") for s0 in starts
                      if nx.has_path(plain, s0, self.goal))
            print(f"route from the spawn: about {full:.0f} steps with shortcuts (looseness {LINK_LOOSENESS}) vs {rec:.0f} along one recording")
        except Exception as exc:
            print("route report skipped:", exc)

    def setup(self, info) -> None:
        super().setup(info)
        try:
            self._setup_goal(info)
        finally:
            self._report_route()

    def _setup_goal(self, info) -> None:
        guess = self.goal
        targets = list(getattr(info, "targets", []) or [])
        if not targets:
            return
        key = hashlib.md5(b"".join(cv2.imencode(".png", t)[1].tobytes() for t in targets)
                          + f"{len(self.frames)}-{LINKS_VERSION}".encode()).hexdigest()[:16]
        cache = self._cache_dir / f"goal_vote_{key}.json"
        try:
            if cache.exists():
                node = int(json.loads(cache.read_text())["goal"])
                print(f"goal: node {node} (from cache; VLAD alone had guessed {guess})")
                if node >= 0:
                    self.goal = node
                return
            node, _ = self._goal_by_voting(targets)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"goal": node}))
        except Exception as exc:                                  # never let the vote stop the run
            print("goal vote failed:", exc)
            return
        if node >= 0:
            self.goal = node
            print(f"goal: node {node} by voting (VLAD alone had guessed {guess})")
        else:
            print(f"goal vote not convincing, keeping the VLAD guess {guess}")



# the tracker reads these module-level values at run time
T.BACKWARD_COST = ROUTE_BACKWARD_COST
T.LINK_RETRIES = ROUTE_LINK_RETRIES
T.LAYOUT_MIN_BINS = _STRIPS
T.STEER_ENABLED = False     # route hints come from the verified route itself (corner countdown), not from one far frame


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", help="exploration data directory (default: download)")
    parser.add_argument("--subsample", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size")
    parser.add_argument("--top-k", type=int, default=100, help="unused (kept for the same command line)")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = RouteAgent(
        cli.exploration_data(args, args.data),
        n_clusters=args.n_clusters,
        subsample=args.subsample,
        top_k_shortcuts=args.top_k,
        cache_dir=Path("cache") / challenge.replace(":", "-"),
    )
    run(agent, challenge, **cli.run_options(args))
