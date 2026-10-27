"""
nav_agent: you drive; it shows ONE route the explorer actually drove (spawn -> goal) and where you are on it.

    uv run source/nav_agent.py --local 21        # rehearse (first start on a maze builds the cache: ~3-5 min)
    uv run source/nav_agent.py                   # challenge (spends an attempt)

Before connecting it builds (and caches) the route map from the exploration data: features, the turn between
consecutive frames, and same-pose links (two frames from the SAME spot -> switch drives by turning on the spot).
No link is ever made from "I can see that place", so gaps you can see through but not drive through never
become part of a route. Goal: the candidates are shown (bottom strip + logs/goal_candidates.png); press 1-5.

On screen: the hint line (one instruction, held steady), and the strip: YOU ARE HERE (route frame you match),
NEXT TURN (route frame where the next turn starts), AFTER THE TURN, GOAL. If the hint says NOT SURE, turn slowly
on the spot until the 'you are here' picture matches. Near the goal: let go -> phase B aligns -> READY -> SPACE.
"""

from __future__ import annotations

import os
import pickle
import textwrap
import time
import traceback
from pathlib import Path

import cli
import cv2
import numpy as np
import goal_picker as GP
import tr_map as TM
from drift_trim import DriftTrim
from final_approach import FinalApproach, intrinsics
from keyboard_agent import KeyboardAgent
from tr_follow import Follower
from vis_nav_sdk import WAIT, Action, SessionInfo, run

LOGS = Path("logs")
SECOND_SPACE_S = 3.0
HUD_WIDTH = 54           # characters the viewer's agent panel shows per line (longer lines were cut off)
GOAL_FAR = 60            # route frames: phase B may not take over while the route is sure the goal is this far
TRUST_KEY = "g"          # hold G: let phase B take over anyway (if the route itself is wrong)
HERE = (76, 209, 132)
TURN = (178, 140, 255)
AFTER = (108, 180, 255)
GOALC = (245, 177, 76)


def report(where: str) -> None:
    text = f"--- error in {where} ---\n{traceback.format_exc()}"
    print(text, flush=True)
    LOGS.mkdir(exist_ok=True)
    with (LOGS / "error.txt").open("a") as f:
        f.write(text)


def _act(a):
    return a[0] if isinstance(a, tuple) and a else a


def _is(a, target) -> bool:
    a = _act(a)
    return a is target or a == target


class NavAgent(KeyboardAgent):
    def __init__(self, data_dir: Path, cache_dir: Path, *, handoff: int = 8, probe: bool = False,
                 confirm: bool = True, guard_space: bool = True, open_picture: bool = True,
                 learn: bool = True, trim: bool = True) -> None:
        self.trim_enabled = trim
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.m = TM.build(self.data_dir, self.cache_dir)
        self.file_views: list[np.ndarray] = []
        tj = self.data_dir / "target.jpg"
        if tj.exists():                     # score the goal photos now, before connecting (the session clock runs later)
            try:
                self.file_views = TM.split_strip(cv2.imread(str(tj)))
                TM.score_goal(self.m, self.file_views, self.cache_dir)
            except Exception:
                report("pre-scoring target.jpg (will score at setup instead)")
        self.learned = 0
        if learn:                           # attempts 2-3: your earlier drives of this maze join the map
            try:
                runs = sorted(Path("runs").glob("nav_*.pkl"))
                if runs:
                    self.learned = TM.add_runs(self.m, runs, self.cache_dir)
                    if self.learned:
                        print(f"nav: learned from {self.learned} earlier drive(s) of this maze -- routes may use them")
            except Exception:
                report("learning from earlier drives (skipped)")
        self.handoff, self.probe, self.confirm, self.guard_space = handoff, probe, confirm, guard_space
        self.open_picture = open_picture
        self.tag, self.log_name = "nav", "phase_b.csv"
        self._reset_state()

    def _reset_state(self) -> None:
        self.follow: Follower | None = None
        self.final: FinalApproach | None = None
        self.targets: list[np.ndarray] = []
        self.cands: list[GP.Candidate] = []
        self.chosen = -1
        self.choosing = False
        self.digit_down = False
        self.picker_broken = False
        self.choice_file: Path | None = None
        self.K = TM.K_DEFAULT
        self._obs = None
        self._seen = -1
        self._hint = "waiting for the first frame"
        self._panel: list = []
        self._tick = 0
        self._b_move = None
        self._engaged = False
        self._gave_up = False
        self._ready: str | None = None
        self._best = 0
        self._space_down = False
        self._space_armed = 0.0
        self._notice_b = ""
        self._ms = 0.0
        self._rec = None
        self._last_out = None
        self._far_warn = False
        self.trim = DriftTrim(enabled=getattr(self, "trim_enabled", True))

    # ------------------------------------------------------------------ setup: goal candidates, route, phase B
    def setup(self, info: SessionInfo) -> None:
        self._reset_state()
        targets = list(getattr(info, "targets", []) or [])
        if not targets or any(GP.is_noise(t) for t in targets):
            print("nav: setup with no real goal photos (the SDK's pre-flight check) -- skipped")
            return
        self.targets = targets
        try:
            self.K = intrinsics(info.camera, targets[0].shape)
        except Exception:
            self.K = TM.K_DEFAULT
        try:
            same = (len(self.file_views) == len(targets) and self.m.goal_scores is not None and all(
                a.shape == b.shape and float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()) < 12
                for a, b in zip(self.file_views, targets)))
            if not same:
                print("nav: scoring the session's goal photos against the exploration frames ...", flush=True)
                TM.score_goal(self.m, targets, self.cache_dir)
            self._setup_candidates()
        except Exception:
            report("goal candidates")
        try:
            self.final = FinalApproach(self.K, targets)
            LOGS.mkdir(exist_ok=True)
            for k, view in enumerate(targets):
                cv2.imwrite(str(LOGS / f"goal_view_{k}.png"), view)
            print("phase B ready" + (" (probe: passive)" if self.probe else ""), flush=True)
        except Exception:
            report("setup phase B (disabled; route hints still work)")

    def _setup_candidates(self) -> None:
        raw = self.m.goal_candidates(5)
        self.cands = [GP.Candidate(g, f"#{k + 1}, score {sc:.0f}", v, inl, 0.0) for k, (g, v, inl, sc) in enumerate(raw)]
        if not self.cands:
            print("nav: no goal candidates found -- drive by the goal photos; phase B still finishes")
            return
        self.choice_file = self.cache_dir / f"nav_choice_{self.m.goal_key}.txt"
        pick = 0
        try:
            prev = int(self.choice_file.read_text())
            pick = next((i for i, c in enumerate(self.cands) if abs(c.node - prev) <= 3), 0)
            print(f"nav: starting on candidate {pick + 1} (your choice last time)")
        except Exception:
            pass
        self._choose(pick, save=False)
        self.choosing = True
        print("goal candidates (press 1-%d; the real goal shows the same NEIGHBOURING panels, not just one poster):"
              % len(self.cands))
        for i, c in enumerate(self.cands):
            r = self.m.records[c.node]
            print(f"   {i + 1}: traj_{r.drive}/{r.index}  ~ goal {TM.VIEW_NAMES[c.view]} view, {c.inliers} inliers, "
                  f"route {self._route_len(c.node):.0f} ticks")
        self._write_montage(open_it=self.open_picture)

    def _route_len(self, g: int) -> float:
        return self.m.route(g)[1]

    def _choose(self, i: int, save: bool = True) -> None:
        c = self.cands[i]
        self.chosen = i
        route, ticks = self.m.route(c.node)
        self.follow = Follower(self.m, route, self.K)
        jumps = sum(1 for u, v in zip(route, route[1:]) if abs(u - v) != 1)
        print(f"nav: goal = candidate {i + 1}; route {len(route)} frames, ~{ticks:.0f} ticks, {jumps} drive switches",
              flush=True)
        if save and self.choice_file is not None:
            try:
                self.choice_file.write_text(str(c.node))
            except Exception:
                pass

    def _image(self, g: int) -> np.ndarray:
        img = cv2.imread(self.m.records[g].path)
        return img if img is not None else np.zeros((240, 320, 3), np.uint8)

    def _write_montage(self, open_it: bool = False) -> None:
        try:
            LOGS.mkdir(exist_ok=True)
            path = LOGS / "goal_candidates.png"
            cv2.imwrite(str(path), GP.montage(self.cands, self.targets, self._image, self.chosen))
            if open_it and hasattr(os, "startfile"):
                os.startfile(str(path.resolve()))
        except Exception:
            report("goal candidates picture")

    # ------------------------------------------------------------------ per step
    def act(self, obs):
        self._obs = obs
        if self.cands and not self.picker_broken:
            try:
                self._keys()
            except Exception:
                self.picker_broken = True
                report("number keys (disabled for this run)")
        if GP.is_noise(obs.image):
            return WAIT
        if obs.step != self._seen:
            self._seen = obs.step
            if self.follow is not None:
                t0 = time.perf_counter()
                try:
                    name = getattr(_act(self._last_out), "name", "") or ""
                    n = self._last_out[1] if isinstance(self._last_out, tuple) and len(self._last_out) > 1 else 1
                    moved = n if ("FORWARD" in name or "BACKWARD" in name) else 0
                    key_turn = (3.44 * n if "LEFT" in name else -3.44 * n if "RIGHT" in name else 0.0)
                    self.follow.localise(obs.image, moved_ticks=moved, key_turn=key_turn)
                    self.trim.observe(self.follow.last_dyaw, name, n)
                    self._hint = self.follow.stable_hint()
                    self._panel = self._route_panel()
                except Exception:
                    report("route follower")
                self._ms = (time.perf_counter() - t0) * 1000
        action = super().act(obs)
        action = self._trim(action)
        was_ready = self._ready is not None
        try:
            out = self._phase_b(action)
        except Exception:
            report("phase B (disabled for the rest of the run)")
            self.final = None
            out = action
        out = self._space_guard(action, out, was_ready)
        self._record(obs, out)
        self._last_out = out
        return out

    def _record(self, obs, out) -> None:
        """runs/nav_<time>.pkl: every new frame + what was sent (for the attempts 2-3 route sheet)."""
        if obs.step == getattr(self, "_rec_step", None) or not self.targets:
            return
        self._rec_step = obs.step
        try:
            if self._rec is None:
                Path("runs").mkdir(exist_ok=True)
                self._rec = open(Path("runs") / time.strftime("nav_%Y%m%d_%H%M%S.pkl"), "wb")
                pickle.dump({"header": True, "data": str(self.data_dir),
                             "goal": self.cands[self.chosen].node if self.cands else None}, self._rec)
            a = _act(out)
            pickle.dump({"step": obs.step, "key": getattr(a, "name", str(a)),
                         "count": out[1] if isinstance(out, tuple) and len(out) > 1 else 1,
                         "route_p": self.follow.p if self.follow else None, "hint": self._hint,
                         "jpg": cv2.imencode(".jpg", obs.image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()}, self._rec)
            self._rec.flush()
        except Exception:
            if not getattr(self, "_rec_warned", False):
                self._rec_warned = True
                report("run recording (the run continues)")

    def _trim(self, action):
        """Cancel the server's forward drift on plain 1-tick FORWARD presses (FORWARD -> FORWARD+turn, same tick)."""
        try:
            if (isinstance(action, tuple) and len(action) > 1 and action[1] == 1
                    and _act(action) == Action.FORWARD):
                side = self.trim.correct(False, False)
                if side:
                    return (Action.FORWARD | getattr(Action, side), 1)
        except Exception:
            if not getattr(self, "_trim_warned", False):
                self._trim_warned = True
                report("drift trim (disabled for this run)")
            self.trim.enabled = False
        return action

    def _keys(self) -> None:
        keys = self.viewer.keys() if getattr(self, "viewer", None) is not None else None
        if keys is None:
            return
        d = GP.digit_of_any(keys)
        if d is not None and not self.digit_down:
            if d <= len(self.cands) and d - 1 != self.chosen:
                self._choose(d - 1)
                self._write_montage()
            elif d - 1 == self.chosen:
                print(f"(candidate {d} is already the goal)", flush=True)
            elif d > len(self.cands):
                print(f"no candidate {d} (there are {len(self.cands)})", flush=True)
        self.digit_down = d is not None
        if self.choosing and any(k in keys for k in ("up", "down", "left", "right")):
            self.choosing = False

    # ------------------------------------------------------------------ phase B (as in final_route_agent)
    def _phase_b(self, action):
        obs = self._obs
        if self.final is None or obs is None:
            return action
        keys_held = not (action is WAIT or action == WAIT)
        if obs.step != getattr(self, "_measured_step", -1):
            self._measured_step = obs.step
            self._tick += 1
            near = self.follow is not None and self.follow.remaining_records() <= self.handoff
            locked = self._engaged or (self.final.last is not None and self.final.last.inliers >= self.final.p.min_inliers)
            self._b_move = None
            self._gave_up = False
            if near or locked or self._tick % 6 == 0 or self.follow is None:
                self._b_move = self.final.step(obs.image, obs.steps_left)
                self._gave_up = self._b_move is None
                est = self.final.last
                if est is not None and est.inliers > self._best:
                    self._best = est.inliers
                    cv2.imwrite(str(LOGS / "best_live.png"), obs.image)
            else:
                self.final.last = None
        route_far = (self.follow is not None and self.follow.fix is not None
                     and self.follow.remaining_records() > GOAL_FAR and not self._trust_key())
        sees = self.final.sees_goal(self.final.last)
        self._far_warn = bool(sees and route_far)
        if sees and not route_far:
            self._engaged = True
        if route_far:
            self._engaged = False
        if keys_held:
            self.final.reset_budget()
            self._ready = None
        if keys_held or self._gave_up:
            self._engaged = False
        execute = self._b_move is not None and not keys_held and not self.probe and self._engaged
        if not execute:
            self.final._expect = None
        if execute and self.confirm and _is(self._b_move, Action.CHECKIN):
            est = self.final.last
            if self._ready is None:
                self._ready = (f"READY -- press SPACE to check in (heading {est.yaw_front:+.1f} deg, front view "
                               f"{est.inliers} inliers). Not close enough? nudge with the arrows, then let go again")
                print("phase B:", self._ready, flush=True)
            self._b_move = None
            return action
        if execute:
            move, self._b_move = self._b_move, None
            self.final.charge(move)
            return move
        return action

    def _trust_key(self) -> bool:
        try:
            keys = self.viewer.keys()
            return TRUST_KEY in keys or TRUST_KEY.upper() in keys
        except Exception:
            return False

    def _space_guard(self, keyboard, out, was_ready: bool):
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

    # ------------------------------------------------------------------ screen
    def _route_panel(self) -> list:
        f = self.follow
        if f is None:
            return []
        i = f.p
        tiles = [(self._image(f.route[i]), f"YOU ARE HERE (route {i}/{len(f.route) - 1})", HERE)]
        t = f.next_turn(i)
        if t is not None:
            deg = f.turn_after(t, i)
            tiles.append((self._image(f.route[t]), f"NEXT TURN {'LEFT' if deg > 0 else 'RIGHT'} {abs(deg):.0f} deg "
                                                   f"starts here", TURN))
            k = f._ahead(t, 0.3)
            tiles.append((self._image(f.route[k]), "...after the turn", AFTER))
        tiles.append((self._image(f.route[-1]), "GOAL (route end)", GOALC))
        return tiles

    def panel(self):
        if self.choosing and self.cands:
            tiles = []
            for i, c in enumerate(self.cands):
                tiles.append((self._image(c.node), f"candidate {i + 1}{' (chosen)' if i == self.chosen else ''}",
                              GOALC if i == self.chosen else HERE))
                if c.view is not None and c.view < len(self.targets):
                    tiles.append((self.targets[c.view], f"goal {TM.VIEW_NAMES[c.view]} view", GOALC))
            return tiles
        return self._panel

    def hud(self) -> list[str]:
        out = []
        for line in self._hud_lines():
            parts = textwrap.wrap(line, HUD_WIDTH, subsequent_indent="    ", break_on_hyphens=False) or [""]
            out += parts
        return out

    def _hud_lines(self) -> list[str]:
        lines = []
        if self.cands and self.choosing:
            lines.append(f">>> GOAL CANDIDATES in the bottom strip (not your position): press 1-{len(self.cands)} "
                         f"(now {self.chosen + 1}), arrows to start")
        if self._far_warn and not self._ready:
            lines.append(f">>> LOOK-ALIKE? Goal-like view, but the route says the goal is still far "
                         f"(route {self.follow.p}/{len(self.follow.route) - 1}). Compare the neighbouring panels. "
                         f"If it IS the goal: hold G and let go")
        elif self._ready:
            lines.append(f">>> {self._ready}")
        elif self._space_armed and time.monotonic() - self._space_armed <= SECOND_SPACE_S:
            lines.append(f">>> {self._notice_b}")
        elif self.follow is not None:
            lines.append(f">>> {self._hint}")
        lines += super().hud()
        f = self.follow
        if f is not None:
            fx = f.fix
            lines.append(f"route {f.p}/{len(f.route) - 1} | ~{f.remaining_ticks():.0f} ticks left | "
                         + (f"match {fx.inliers} | view off {fx.yaw:+.0f} deg" if fx else f"no match ({f.misses})")
                         + f" | {self._ms:.0f} ms" + (f" | goal candidate {self.chosen + 1} (1-{len(self.cands)} to switch)"
                                                       if self.cands else ""))
        lines.append(self.trim.status())
        if self.final is not None:
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
    parser.add_argument("--handoff", type=int, default=8, help="phase B measures every frame within this many route frames")
    parser.add_argument("--probe", action="store_true", help="phase B measures only, never moves")
    parser.add_argument("--auto-checkin", action="store_true", help="phase B checks in by itself")
    parser.add_argument("--single-space", action="store_true", help="disable the double-SPACE guard")
    parser.add_argument("--no-open", action="store_true", help="don't open logs/goal_candidates.png automatically")
    parser.add_argument("--no-learn", action="store_true", help="ignore earlier drives in runs/ (exploration data only)")
    parser.add_argument("--no-trim", action="store_true", help="don't cancel the measured forward drift")
    args = parser.parse_args()
    challenge = cli.challenge(args)
    data = cli.exploration_data(args, args.data)
    agent = NavAgent(data, Path("cache") / challenge.replace(":", "-"), handoff=args.handoff, probe=args.probe,
                     confirm=not args.auto_checkin, guard_space=not args.single_space, open_picture=not args.no_open,
                     learn=not args.no_learn, trim=not args.no_trim)
    run(agent, challenge, **cli.run_options(args))
