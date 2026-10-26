"""
Phase B test harness: the baseline agent, with the last-metre controller on top.

You drive with the arrows exactly as in baseline_agent.py. Phase B measures each new frame
once. It takes over when you are not touching the keys AND either its front-view match is
strong (FinalApproach.sees_goal). (--handoff hops only sets how often it measures: the hop count
is unreliable near goals, and engaging on it made phase B spin while you were still driving.) With no lock it only measures every --every frames,
to stay smooth. Any arrow, space or escape hands control straight back to you.

    uv run source/final_agent.py --local 7            # normal
    uv run source/final_agent.py --local 7 --probe    # passive: never moves, only measures
    uv run source/final_agent.py --local 7 --auto-checkin   # phase B checks in without asking

By default phase B turns and drives but does NOT check in: when it would, the status column
shows "READY -- press SPACE" and the decision is yours.

--probe is for finding out where the goal photos start to match: drive around yourself and
watch the "matches per view" line in the status column.

Writes to logs/: phase_b.csv (every step), goal_view_0..3.png, best_live.png (the live frame
with the most inliers so far), error.txt (only if something raised).
"""

from __future__ import annotations

import traceback
from pathlib import Path

import cli
import cv2
import numpy as np
import baseline_agent as _kit
from baseline_agent import BaselineAgent

# Louie's upload changed these kit values; restore the ones phase B was rehearsed with.
# (Module globals: affects this process only, so final_route_agent.py keeps Louie's 250.)
_kit.MIN_SHORTCUT_GAP = 50
_kit.PREVIEW_NODES = 5
from final_approach import (
    FinalApproach,
    GoalMatcher,
    Params,
    intrinsics,
    load_goal_views,
    place_scores,
)
from vis_nav_sdk import WAIT, Action, Observation, SessionInfo, run

LOGS = Path("logs")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):
        return it


def report(where: str) -> None:
    """Print the current exception and keep a copy: the SDK may close the window before you
    can read the terminal."""
    text = f"--- error in {where} ---\n{traceback.format_exc()}"
    print(text, flush=True)
    LOGS.mkdir(exist_ok=True)
    with (LOGS / "error.txt").open("a") as f:
        f.write(text)


class FinalAgent(BaselineAgent):
    def __init__(self, data_dir: Path, *, handoff_hops: int = 5, probe: bool = False, confirm: bool = True,
                 use_feature_goal: bool = False, measure_every: int = 8, **kwargs) -> None:
        super().__init__(data_dir, **kwargs)
        self.feature_goal_node: int | None = None
        if use_feature_goal:
            try:
                self.feature_goal_node = self._feature_goal(Path(data_dir), Path(kwargs.get("cache_dir", "cache")))
            except Exception:
                report("feature goal (falling back to the baseline's VLAD goal)")
        self.handoff_hops = handoff_hops
        self.measure_every = measure_every
        self._tick = 0
        self._measured_step = -1
        self._move = None
        self._engaged = False
        self._gave_up = False
        self.probe = probe
        self.confirm = confirm  # phase B aligns but YOU press space to check in
        self._ready: str | None = None
        self.final: FinalApproach | None = None
        self._best = 0
        self.tag = "manual"
        self.log_name = "phase_b.csv"

    def _feature_goal(self, data_dir: Path, cache_dir: Path) -> int | None:
        """The goal node from local features over ALL the goal views, not VLAD on the front one:
        the node whose surrounding stretch of path sees the most of them."""
        views = load_goal_views(data_dir)
        if not views:
            print(f"feature goal: no target*.jpg in {data_dir}; keeping the VLAD goal")
            return None
        matcher = GoalMatcher(np.eye(3), views, Params())
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "goal_counts.npy"
        counts = None
        if cache_file.exists():
            cached = np.load(cache_file)
            if cached.shape == (len(self.frames), len(views)):
                counts = cached
        if counts is None:
            rows = []
            for fr in tqdm(self.frames, desc="goal views"):
                img = cv2.imread(str(fr.path))
                rows.append(matcher.raw_counts(img) if img is not None else [0] * len(views))
            counts = np.array(rows)
            np.save(cache_file, counts)
        scores = place_scores(counts, [fr.trajectory for fr in self.frames])
        best = np.flatnonzero(scores == scores.max())
        node = int(best[len(best) // 2])
        lo, hi = max(0, node - 4), min(len(counts), node + 5)
        peak = [(int(lo + counts[lo:hi, v].argmax()), int(counts[lo:hi, v].max())) for v in range(len(views))]
        print(f"feature goal: node {node} (score {scores[node]:.0f}); best node:count per view {peak}")
        return node

    def setup(self, info: SessionInfo) -> None:
        super().setup(info)
        if self.feature_goal_node is not None:
            print(f"goal node: {self.goal} (VLAD, front view) -> {self.feature_goal_node} (local features, all views)")
            self.goal = self.feature_goal_node
        print("camera:", info.camera)
        print("limits:", info.limits)
        K = intrinsics(info.camera, info.targets[0].shape)
        print("K =\n", K)
        self.final = None
        self._best = 0
        self._measured_step = -1
        self._move = None
        self._engaged = False
        self._gave_up = False
        self._ready = None
        try:
            self.final = FinalApproach(K, info.targets)
            LOGS.mkdir(exist_ok=True)
            for k, view in enumerate(info.targets):
                cv2.imwrite(str(LOGS / f"goal_view_{k}.png"), view)
            print("phase B ready" + (" (probe: passive)" if self.probe else ""), flush=True)
        except Exception:
            report("setup (phase B disabled, baseline still works)")

    def act(self, obs: Observation):
        try:
            action = super().act(obs)  # localises, then reads the keyboard
        except Exception:
            report("act: baseline / keyboard")
            raise
        if self.final is None:
            return action
        keys_held = not (action is WAIT or action == WAIT)
        if obs.step != self._measured_step:  # a new frame: measure it once (act repeats while you wait)
            self._measured_step = obs.step
            self._tick += 1
            near = len(self.path) - 1 <= self.handoff_hops
            locked = self._engaged or (self.final.last is not None and self.final.last.inliers >= self.final.p.min_inliers)
            self._move = None
            self._gave_up = False
            if near or locked or self._tick % self.measure_every == 0:  # far and blind: only every Nth frame
                try:
                    self._move = self.final.step(obs.image, obs.steps_left)
                    self._gave_up = self._move is None
                except Exception:
                    report("act: phase B (disabled for the rest of the run)")
                    self.final = None
                    return action
                est = self.final.last
                if est is not None and est.inliers > self._best:
                    self._best = est.inliers
                    cv2.imwrite(str(LOGS / "best_live.png"), obs.image)
                if self.final.rows:
                    self.final.rows[-1].update(hops=len(self.path) - 1, keys=int(keys_held))
            else:
                self.final.last = None
        near = len(self.path) - 1 <= self.handoff_hops
        if self.final.sees_goal(self.final.last):
            self._engaged = True  # latched: phase B keeps control through its look-around scans
        if keys_held:
            self.final.reset_budget()  # you drove: phase B gets a fresh budget next time
            self._ready = None
        if keys_held or self._gave_up:
            self._engaged = False  # you took over, or phase B had nothing to do (scanned a full circle)
        engaged = self._engaged  # only on a confirmed goal match: the hop count misfired near goals
                                 # (maze 314: ~100 ticks of scanning while you were still driving)
        execute = self._move is not None and not keys_held and not self.probe and engaged
        if not execute:
            self.final._expect = None  # phase B's probe turn didn't happen: don't judge the next frame by it
        if execute and self.final.rows:
            self.final.rows[-1]["executed"] = 1
        if execute and self.confirm and (self._move is Action.CHECKIN or self._move == Action.CHECKIN):
            # confirm mode: phase B is aligned and would check in -- leave that decision to you
            est = self.final.last
            if self._ready is None:
                self._ready = (f"READY -- press SPACE to check in "
                               f"(heading {est.yaw_front:+.1f} deg, front view {est.inliers} inliers). "
                               f"Not close enough? nudge with the arrows, then let go again")
                print("phase B:", self._ready, flush=True)
            if self.final.rows:
                self.final.rows[-1]["executed"] = 0
                self.final.rows[-1]["note"] = "READY (waiting for space)"
            self._move = None
            return action
        if execute:
            move, self._move = self._move, None  # one move per frame
            self.final.charge(move)
            return move
        return action

    def hud(self) -> list[str]:
        lines = super().hud()
        if self.final is not None:
            if self._ready:
                lines.append(f">>> {self._ready}")
            lines.append(f"phase B{' (probe)' if self.probe else ''}: {self.final.status}")
        return lines

    def finish(self, result) -> None:
        print("result:", result)
        LOGS.mkdir(exist_ok=True)
        with (LOGS / "results.txt").open("a") as f:
            f.write(f"{self.tag}\t{result}\n")
        if self.final is not None:
            self.final.dump_csv(LOGS / self.log_name)
            print(f"wrote logs/{self.log_name}")


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", help="exploration data directory (default: download)")
    parser.add_argument("--subsample", type=int, default=5, help="keep every Nth frame")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size")
    parser.add_argument("--top-k", type=int, default=30, help="visual shortcut edges")
    parser.add_argument("--handoff", type=int, default=5, help="hand over at this many hops")
    parser.add_argument("--probe", action="store_true", help="measure only, never move")
    parser.add_argument("--auto-checkin", action="store_true",
                        help="let phase B check in by itself (default: it says READY and you press space)")
    parser.add_argument("--feature-goal", action="store_true",
                        help="EXPERIMENTAL: pick the goal node from local features (picked a wrong place on maze 7)")
    parser.add_argument("--every", type=int, default=8, help="far from the goal, match every Nth frame")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = FinalAgent(
        cli.exploration_data(args, args.data),
        handoff_hops=args.handoff,
        probe=args.probe,
        confirm=not args.auto_checkin,
        use_feature_goal=args.feature_goal,
        measure_every=args.every,
        n_clusters=args.n_clusters,
        subsample=args.subsample,
        top_k_shortcuts=args.top_k,
        cache_dir=Path("cache") / challenge.replace(":", "-"),
    )
    run(agent, challenge, **cli.run_options(args))
