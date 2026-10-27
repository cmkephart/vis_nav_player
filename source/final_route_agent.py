"""
Final agent: Louie's route hints (phase A, you drive) + the phase B last-metre controller.

You drive with the arrows following the route hints (verified links, goal found by voting).
Phase B measures each new frame once and takes over only when you are NOT touching the keys AND
it has a confirmed goal match. Any key hands control straight back to you.

Confirm mode (default): phase B aligns, then shows "READY -- press SPACE". SPACE is guarded:
before READY, a single press does nothing; press it a SECOND time within 3 s to check in anyway.
The route agent's own "press space" hints are replaced by "let go, wait for READY".

    uv run source/final_route_agent.py --local 21             # rehearse
    uv run source/final_route_agent.py                        # challenge (spends an attempt)
    uv run source/final_route_agent.py --local 21 --probe     # phase B measures only
    uv run source/final_route_agent.py --local 21 --no-pick   # skip the goal candidate picker

Goal candidates (goal_picker.py): at the start the panel strip shows each candidate goal frame beside
the goal view it matches, and logs/goal_candidates.png opens. Press 1..5 to choose (works any time);
start driving with the arrows. The choice is remembered for the next attempt.
    $env:LINK_LOOSENESS=2                                     # more route shortcuts (riskier)

Layout: this file in source/, Louie's stable files in source/stable/. The stable folder goes
FIRST on sys.path so its tracking_agent.py / baseline_agent.py win over any other copies.
Logs: logs/ (phase B), runs/ + tracking_log.csv (route tracker).
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "stable"))  # BEFORE the imports below

import os  # noqa: E402

import cli  # noqa: E402
import cv2  # noqa: E402
import goal_picker as GP  # noqa: E402
from baseline_agent import GOAL, MATCH  # noqa: E402
from final_approach import FinalApproach, GoalMatcher, Params, intrinsics  # noqa: E402
from route_agent import RouteAgent  # noqa: E402  (pulls stable tracking_agent + baseline_agent)
import route_agent; route_agent.GOAL_MIN_CLUSTER = 20  # accept a clear-winner goal vote
from vis_nav_sdk import WAIT, Action, SessionInfo, run  # noqa: E402

LOGS = Path("logs")
SECOND_SPACE_S = 3.0  # an unguarded check-in needs a second SPACE press within this many seconds


def report(where: str) -> None:
    text = f"--- error in {where} ---\n{traceback.format_exc()}"
    print(text, flush=True)
    LOGS.mkdir(exist_ok=True)
    with (LOGS / "error.txt").open("a") as f:
        f.write(text)


def _act(a):
    """The Action inside a keyboard result (which may be a bare Action or an (Action, count) tuple)."""
    return a[0] if isinstance(a, tuple) and a else a


def _is(a, target) -> bool:
    a = _act(a)
    return a is target or a == target


class FinalRouteAgent(RouteAgent):
    def __init__(self, data_dir: Path, *, handoff_hops: int = 5, probe: bool = False, confirm: bool = True,
                 measure_every: int = 8, guard_space: bool = True, pick: bool = True,
                 open_picture: bool = True, **kwargs) -> None:
        self.pick = pick
        self.open_picture = open_picture
        self._cands: list[GP.Candidate] = []
        self._chosen = -1
        self._choosing = False
        self._digit_down = False
        self._keys_info_shown = False
        self._picker_broken = False
        self._cand_targets = []
        self._cand_cache: Path | None = None
        self._vote_top = None
        super().__init__(data_dir, **kwargs)  # builds the route graph (cached after the first run)
        self.handoff_hops = handoff_hops
        self.measure_every = measure_every
        self.probe = probe
        self.confirm = confirm
        self.guard_space = guard_space
        self.final: FinalApproach | None = None
        self._obs = None
        self._tick = 0
        self._measured_step = -1
        self._b_move = None          # NOT self._move: the tracker uses that name for its hint
        self._engaged = False
        self._gave_up = False
        self._ready: str | None = None
        self._best = 0
        self._space_down = False
        self._space_armed = 0.0
        self._notice_b = ""
        self.tag = "route+B"
        self.log_name = "phase_b.csv"

    # -- setup: route agent (graph, goal vote) first, then phase B
    def setup(self, info: SessionInfo) -> None:
        super().setup(info)
        print("camera:", info.camera)
        print("limits:", info.limits)
        self.final = None
        self._best = 0
        self._measured_step = -1
        self._b_move = None
        self._engaged = False
        self._gave_up = False
        self._ready = None
        self._space_down = False
        self._space_armed = 0.0
        self._notice_b = ""
        try:
            K = intrinsics(info.camera, info.targets[0].shape)
            print("K =\n", K)
            self.final = FinalApproach(K, info.targets)
            LOGS.mkdir(exist_ok=True)
            for k, view in enumerate(info.targets):
                cv2.imwrite(str(LOGS / f"goal_view_{k}.png"), view)
            print("phase B ready" + (" (probe: passive)" if self.probe else ""), flush=True)
        except Exception:
            report("setup (phase B disabled, route hints still work)")

    # -- goal candidates (goal_picker.py)
    def _goal_by_voting(self, targets):
        node, top = super()._goal_by_voting(targets)
        self._vote_top = top
        return node, top

    def _setup_goal(self, info) -> None:
        vlad = self.goal                 # the baseline's VLAD guess, before the vote replaces it
        self._vote_top = None
        super()._setup_goal(info)
        self._cands, self._chosen, self._choosing = [], -1, False
        if not self.pick:
            return
        try:
            self._setup_candidates(info, vlad)
        except Exception:
            report("goal candidates (picker disabled, the vote's goal is kept)")
            self._cands = []

    def _setup_candidates(self, info, vlad) -> None:
        targets = list(getattr(info, "targets", []) or [])
        if not targets:
            return
        if any(GP.is_noise(t) for t in targets):
            print("goal candidates: skipped (these goal views are random static -- the SDK's pre-flight check)")
            return
        self._cand_targets = targets
        key = GP.cache_key(targets, len(self.frames), route_agent.LINKS_VERSION)
        self._cand_cache = self._cache_dir / f"goal_candidates_{key}.json"
        cached = GP.load_cache(self._cand_cache)
        top = self._vote_top if self._vote_top is not None else cached.get("top")
        if top is None:                  # the vote came from route_agent's cache, which keeps only the winner
            print("goal candidates: re-running the goal vote once to list the runners-up ...", flush=True)
            _, top = self._goal_by_voting(targets)
        K = intrinsics(info.camera, targets[0].shape)
        matcher = GoalMatcher(K, targets, Params())
        self._cands = GP.build(top, vlad, self.frames, self._image, matcher)
        if not self._cands:
            return
        prev = cached.get("choice")
        pick = next((i for i, c in enumerate(self._cands) if prev is not None and abs(c.node - prev) <= GP.REFINE), None)
        why = "your choice last time"
        if pick is None:
            pick = next((i for i, c in enumerate(self._cands) if abs(c.node - (self.goal or -999)) <= GP.REFINE), 0)
            why = "the vote's pick" if self._cands[pick].source.startswith("vote") else "the VLAD guess"
        self._choose(pick, save=False)
        GP.save_cache(self._cand_cache, top, prev)
        self._choosing = True
        print("goal candidates:")
        for i, c in enumerate(self._cands):
            print("   " + c.label(i) + ("   <- current (" + why + ")" if i == pick else ""))
        self._write_montage(open_it=self.open_picture)
        print("press 1..%d to choose; logs/goal_candidates.png shows them beside the goal views" % len(self._cands),
              flush=True)

    def _write_montage(self, open_it: bool = False) -> None:
        try:
            LOGS.mkdir(exist_ok=True)
            path = LOGS / "goal_candidates.png"
            cv2.imwrite(str(path), GP.montage(self._cands, self._cand_targets, self._image, self._chosen))
            if open_it and hasattr(os, "startfile"):
                os.startfile(str(path.resolve()))  # Windows: opens in the default image viewer
        except Exception:
            report("goal candidates picture")

    def _choose(self, i: int, save: bool = True) -> None:
        c = self._cands[i]
        self._chosen = i
        self.goal = c.node
        self._localised_step = None      # re-plan the route on the next frame
        if save and self._cand_cache is not None:
            cached = GP.load_cache(self._cand_cache)
            GP.save_cache(self._cand_cache, cached.get("top", []), c.node)
            print(f"goal: candidate {i + 1}, node {c.node} (remembered for the next attempt)", flush=True)

    def _picker_keys(self) -> None:
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        keys = viewer.keys()             # a Keys object: supports `in`, not iteration
        if not self._keys_info_shown:
            self._keys_info_shown = True
            print(f"(viewer keys object: {type(keys).__name__}; "
                  f"{[a for a in dir(keys) if not a.startswith('_')]})", flush=True)
        d = GP.digit_of(keys)
        if d is not None and not self._digit_down and self._cands:
            if d <= len(self._cands):
                if d - 1 != self._chosen:
                    self._choose(d - 1)
                    self._write_montage()
            else:
                print(f"no candidate {d} (there are {len(self._cands)})", flush=True)
        self._digit_down = d is not None
        if self._choosing and any(k in keys for k in ("up", "down", "left", "right")):
            self._choosing = False       # driving started: back to the route strip

    def panel(self):
        if not (self._choosing and self._cands):
            return super().panel()
        tiles = []
        for i, c in enumerate(self._cands):
            tiles.append((self._image(c.node), f"candidate {i + 1}{' (chosen)' if i == self._chosen else ''}: node {c.node}",
                          GOAL if i == self._chosen else MATCH))
            if c.view is not None and c.view < len(self._cand_targets):
                tiles.append((self._cand_targets[c.view], f"goal {GP.VIEW_NAMES[c.view]} view", GOAL))
        return tiles

    # -- the tracker calls _decide(keyboard action) and records whatever we return
    def act(self, obs):
        self._obs = obs
        if self._cands and not self._picker_broken:
            try:
                self._picker_keys()
            except Exception:
                self._picker_broken = True   # report once, never every frame
                report("goal picker keys (number keys disabled for this run; the goal stays as it is)")
        return super().act(obs)

    def _decide(self, action):
        was_ready = self._ready is not None  # pressing SPACE clears _ready inside _phase_b
        try:
            out = self._phase_b(action)
        except Exception:
            report("_decide: phase B (disabled for the rest of the run)")
            self.final = None
            out = action
        return self._space_guard(action, out, was_ready)

    def _phase_b(self, action):
        obs = self._obs
        if self.final is None or obs is None:
            return action
        keys_held = not (action is WAIT or action == WAIT)
        if obs.step != self._measured_step:  # a new frame: measure it once
            self._measured_step = obs.step
            self._tick += 1
            near = len(self.path) - 1 <= self.handoff_hops
            locked = self._engaged or (self.final.last is not None
                                       and self.final.last.inliers >= self.final.p.min_inliers)
            self._b_move = None
            self._gave_up = False
            if near or locked or self._tick % self.measure_every == 0:
                self._b_move = self.final.step(obs.image, obs.steps_left)
                self._gave_up = self._b_move is None
                est = self.final.last
                if est is not None and est.inliers > self._best:
                    self._best = est.inliers
                    cv2.imwrite(str(LOGS / "best_live.png"), obs.image)
                if self.final.rows:
                    self.final.rows[-1].update(hops=len(self.path) - 1, keys=int(keys_held))
            else:
                self.final.last = None
        if self.final.sees_goal(self.final.last):
            self._engaged = True
        if keys_held:
            self.final.reset_budget()
            self._ready = None
        if keys_held or self._gave_up:
            self._engaged = False
        execute = self._b_move is not None and not keys_held and not self.probe and self._engaged
        if not execute:
            self.final._expect = None
        if execute and self.final.rows:
            self.final.rows[-1]["executed"] = 1
        if execute and self.confirm and _is(self._b_move, Action.CHECKIN):
            est = self.final.last
            if self._ready is None:
                self._ready = (f"READY -- press SPACE to check in "
                               f"(heading {est.yaw_front:+.1f} deg, front view {est.inliers} inliers). "
                               f"Not close enough? nudge with the arrows, then let go again")
                print("phase B:", self._ready, flush=True)
            if self.final.rows:
                self.final.rows[-1]["executed"] = 0
                self.final.rows[-1]["note"] = "READY (waiting for space)"
            self._b_move = None
            return action
        if execute:
            move, self._b_move = self._b_move, None  # one move per frame
            self.final.charge(move)
            return move  # phase B already returns (Action, n) tuples, or a bare CHECKIN
        return action

    def _space_guard(self, keyboard, out, was_ready: bool):
        """Block a stray SPACE before READY: the first press only arms, a second fresh press within
        SECOND_SPACE_S seconds checks in. Phase B's own check-in (--auto-checkin) is not affected."""
        pressed = _is(keyboard, Action.CHECKIN)
        fresh = pressed and not self._space_down
        self._space_down = pressed
        if not (pressed and self.guard_space and self.confirm and self.final is not None):
            return out
        if was_ready:
            return out
        if fresh and time.monotonic() - self._space_armed <= SECOND_SPACE_S:
            print("check-in WITHOUT phase B READY (second SPACE)", flush=True)
            return out
        if fresh:
            self._space_armed = time.monotonic()
            self._notice_b = "SPACE blocked: phase B is not READY. Press SPACE again within 3 s to check in anyway."
            print(self._notice_b, flush=True)
        return WAIT

    # -- HUD: phase B owns the "press space" decision
    def hud(self) -> list[str]:
        lines = []
        if self._cands:
            n = len(self._cands)
            if self._choosing:
                lines.append(f">>> GOAL CANDIDATES in the bottom strip (not your position): press 1-{n} to choose "
                             f"(now {self._chosen + 1}), arrows to start. "
                             "Real place = same panels in the same order, not just one poster. "
                             "Picture: logs/goal_candidates.png")
            else:
                lines.append(f"goal: candidate {self._chosen + 1} of {n} (node {self.goal}) -- press 1-{n} to switch")
        for line in super().hud():
            if "press space" in line:  # route/baseline hints (lowercase); phase B's says "SPACE"
                line = line.split("press space")[0] + "near the goal: LET GO of the keys and wait for READY"
            lines.append(line)
        if self.final is not None:
            if self._ready:
                lines.append(f">>> {self._ready}")
            elif self._space_armed and time.monotonic() - self._space_armed <= SECOND_SPACE_S:
                lines.append(f">>> {self._notice_b}")
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
        try:
            self._save_store()  # keep the tracker's SIFT cache for the next attempt
        except Exception:
            pass


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", help="exploration data directory (default: download)")
    parser.add_argument("--subsample", type=int, default=1, help="keep every Nth frame (route agent was tested at 1)")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size")
    parser.add_argument("--top-k", type=int, default=100, help="unused by the route agent")
    parser.add_argument("--handoff", type=int, default=5, help="measure every frame within this many hops")
    parser.add_argument("--every", type=int, default=8, help="far from the goal, phase B matches every Nth frame")
    parser.add_argument("--probe", action="store_true", help="phase B measures only, never moves")
    parser.add_argument("--auto-checkin", action="store_true", help="phase B checks in by itself")
    parser.add_argument("--single-space", action="store_true", help="disable the double-SPACE guard")
    parser.add_argument("--no-pick", action="store_true", help="no goal candidate picker (use the vote's goal)")
    parser.add_argument("--no-open", action="store_true", help="don't open logs/goal_candidates.png automatically")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = FinalRouteAgent(
        cli.exploration_data(args, args.data),
        handoff_hops=args.handoff,
        probe=args.probe,
        confirm=not args.auto_checkin,
        measure_every=args.every,
        guard_space=not args.single_space,
        pick=not args.no_pick,
        open_picture=not args.no_open,
        n_clusters=args.n_clusters,
        subsample=args.subsample,
        top_k_shortcuts=args.top_k,
        cache_dir=Path("cache") / challenge.replace(":", "-"),
    )
    run(agent, challenge, **cli.run_options(args))
