"""
Phase B test harness: the baseline agent, with the last-metre controller on top.

You drive with the arrows exactly as in baseline_agent.py. Phase B runs on every frame and
takes over when you are not touching the keys AND either the baseline says you are within
--handoff hops of the goal node, or phase B itself finds enough matches against the goal
photos (the baseline's goal node can be a poor guess). Any arrow, space or escape hands
control straight back to you.

    uv run source/final_agent.py --local 7            # normal
    uv run source/final_agent.py --local 7 --probe    # passive: never moves, only measures

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
from baseline_agent import BaselineAgent
from final_approach import FinalApproach, intrinsics
from vis_nav_sdk import WAIT, Observation, SessionInfo, run

LOGS = Path("logs")


def report(where: str) -> None:
    """Print the current exception and keep a copy: the SDK may close the window before you
    can read the terminal."""
    text = f"--- error in {where} ---\n{traceback.format_exc()}"
    print(text, flush=True)
    LOGS.mkdir(exist_ok=True)
    with (LOGS / "error.txt").open("a") as f:
        f.write(text)


class FinalAgent(BaselineAgent):
    def __init__(self, data_dir: Path, *, handoff_hops: int = 5, probe: bool = False, **kwargs) -> None:
        super().__init__(data_dir, **kwargs)
        self.handoff_hops = handoff_hops
        self.probe = probe
        self.final: FinalApproach | None = None
        self._best = 0

    def setup(self, info: SessionInfo) -> None:
        super().setup(info)
        print("camera:", info.camera)
        print("limits:", info.limits)
        K = intrinsics(info.camera, info.targets[0].shape)
        print("K =\n", K)
        self.final = None
        self._best = 0
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
        try:
            move = self.final.step(obs.image, obs.steps_left)
            est = self.final.last
            if est is not None and est.inliers > self._best:
                self._best = est.inliers
                cv2.imwrite(str(LOGS / "best_live.png"), obs.image)
        except Exception:
            report("act: phase B (disabled for the rest of the run)")
            self.final = None
            return action

        near = len(self.path) - 1 <= self.handoff_hops
        evidence = est is not None and est.inliers >= self.final.p.engage_inliers
        execute = move is not None and not keys_held and not self.probe and (near or evidence)
        if self.final.rows:
            self.final.rows[-1]["executed"] = int(execute)
            self.final.rows[-1]["hops"] = len(self.path) - 1
            self.final.rows[-1]["keys"] = int(keys_held)
        return move if execute else action

    def hud(self) -> list[str]:
        lines = super().hud()
        if self.final is not None:
            lines.append(f"phase B{' (probe)' if self.probe else ''}: {self.final.status}")
        return lines

    def finish(self, result) -> None:
        print("result:", result)
        if self.final is not None:
            self.final.dump_csv(LOGS / "phase_b.csv")
            print("wrote logs/phase_b.csv")


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", help="exploration data directory (default: download)")
    parser.add_argument("--subsample", type=int, default=5, help="keep every Nth frame")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size")
    parser.add_argument("--top-k", type=int, default=30, help="visual shortcut edges")
    parser.add_argument("--handoff", type=int, default=5, help="hand over at this many hops")
    parser.add_argument("--probe", action="store_true", help="measure only, never move")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = FinalAgent(
        cli.exploration_data(args, args.data),
        handoff_hops=args.handoff,
        probe=args.probe,
        n_clusters=args.n_clusters,
        subsample=args.subsample,
        top_k_shortcuts=args.top_k,
        cache_dir=Path("cache") / challenge.replace(":", "-"),
    )
    run(agent, challenge, **cli.run_options(args))
