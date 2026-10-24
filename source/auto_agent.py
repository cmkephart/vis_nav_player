"""
Headless stand-in for the whole pipeline: no keyboard, no viewer window, so it runs on the
HPC nodes. It is NOT the navigation method (that is your partner's job), just enough of one to
get phase B into position:

    far from the goal   follow the baseline's graph: do the move recorded on the path's first edge
    near the goal       final_approach takes over (within --handoff hops, or when it finds
                        enough matches against the goal photos by itself)
    at the goal node    with no lock: turn on the spot looking for the goal, then check in

It checks in once phase B says so, or after --budget steps, whichever comes first. Local runs
are free, so check-ins that miss cost nothing.

    uv run source/auto_agent.py --local 7 --yes --no-browser
    for s in 1 2 3 4 5; do uv run source/auto_agent.py --local $s --yes --no-browser; done

Each run prints `result:` (distance, heading, score as the SDK reports them), appends it to
logs/results.txt, and writes logs/phase_b_<tag>.csv with every decision.
"""

from __future__ import annotations

from pathlib import Path

import cli
from final_agent import FinalAgent, report
from vis_nav_sdk import Action, Observation, SessionInfo, run

MOVES = {"FORWARD", "BACKWARD", "LEFT", "RIGHT"}


class AutoAgent(FinalAgent):
    def __init__(self, data_dir: Path, *, tag: str = "run", budget: int = 4000,
                 search_ticks: int = 40, **kwargs) -> None:
        super().__init__(data_dir, **kwargs)
        self.tag = tag
        self.log_name = f"phase_b_{tag}.csv"
        self.budget = budget
        self.search_ticks = search_ticks
        self._acts = 0
        self._scan = 0

    def setup(self, info: SessionInfo) -> None:
        super().setup(info)
        self._acts = 0
        self._scan = 0

    def _render_panel(self):  # no viewer: skip loading preview images every step
        return []

    def act(self, obs: Observation):
        try:
            if obs.step != self._localised_step:
                self._localise(obs.image)
                self._localised_step = obs.step
            return self._drive(obs)
        except Exception:
            report("auto act")
            raise

    def _drive(self, obs: Observation):
        self._acts += 1
        if self._acts > self.budget or obs.steps_left <= 3:
            return Action.CHECKIN

        hops = len(self.path) - 1
        move, est = None, None
        if self.final is not None and (hops <= self.handoff_hops or self._acts % self.measure_every == 0):
            try:
                move = self.final.step(obs.image, obs.steps_left)
                est = self.final.last
            except Exception:
                report("auto act: phase B (disabled for the rest of the run)")
                self.final = None
        engaged = est is not None and (hops <= self.handoff_hops or self.final.sees_goal(est))
        if self.final is not None and self.final.rows:
            self.final.rows[-1].update(hops=hops, engaged=int(engaged and move is not None))

        if engaged and move is not None:
            self._scan = 0
            return move

        if hops >= 1:  # follow the graph
            action = self._edge_action(self.path[0], self.path[1])
            return (Action[action], 1) if action in MOVES else (Action.LEFT, 1)  # "?" = visual jump: look around

        self._scan += 1  # at the goal node, nothing matches: look around, then give up
        if self._scan > self.search_ticks:
            return Action.CHECKIN
        return (Action.LEFT, 1)


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", help="exploration data directory (default: download)")
    parser.add_argument("--subsample", type=int, default=5, help="keep every Nth frame")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size")
    parser.add_argument("--top-k", type=int, default=30, help="visual shortcut edges")
    parser.add_argument("--handoff", type=int, default=5, help="hand over at this many hops")
    parser.add_argument("--budget", type=int, default=4000, help="check in after this many steps")
    parser.add_argument("--feature-goal", action="store_true", help="EXPERIMENTAL: goal node from local features")
    parser.add_argument("--tag", help="name for this run's log files (default: the challenge)")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = AutoAgent(
        cli.exploration_data(args, args.data),
        tag=args.tag or challenge.replace(":", "-"),
        budget=args.budget,
        handoff_hops=args.handoff,
        use_feature_goal=args.feature_goal,
        n_clusters=args.n_clusters,
        subsample=args.subsample,
        top_k_shortcuts=args.top_k,
        cache_dir=Path("cache") / challenge.replace(":", "-"),
    )
    run(agent, challenge, **cli.run_options(args))
