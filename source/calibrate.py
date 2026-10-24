"""
Calibrate phase B against the truth. Runs the local simulator directly (no SDK, no baseline),
so it knows the robot's real distance and heading to the goal, and logs them next to what
phase B measures on every frame.

    uv run source/calibrate.py            # maze 7
    uv run source/calibrate.py --maze 3

Keys (click the window first; it is a tkinter window, OpenCV here has no GUI):
    arrows or WASD   move by the current tick count       1-9   ticks per key press
    t                turn calibration: 1-tick LEFT turns for a full circle
    g                goal sweep (do it ON the goal): full spin, then backs off and returns
    p                phase B drives from here until it checks in (prints the true result)
    r                20 trials (do it ON the goal): scramble the pose nearby, let phase B recover
    q or esc         quit, write logs/calibrate_<maze>.csv and print a summary

Drive to the goal by eye (the window shows goal view 0 and the true distance), press g
there, and also drive around near it. The summary shows phase B's front-view inliers by true
distance and heading, and where the current Params would check in.
"""

from __future__ import annotations

import argparse
import base64
import csv
import math
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from final_approach import FinalApproach
from vis_nav_sdk import Action

K = np.array([[92.0, 0, 160], [0, 92.0, 120], [0, 0, 1]])
KEYS = {  # tkinter key names
    "up": "FORWARD", "down": "BACKWARD", "left": "LEFT", "right": "RIGHT",
    "w": "FORWARD", "s": "BACKWARD", "a": "LEFT", "d": "RIGHT",
}


class Calibrator:
    def __init__(self, maze: int, seed: int) -> None:
        self.world = sim.Simulator(sim.Textures.find(), maze, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
        self.maze = maze
        self.frame = self.world.initial_frame()
        self.targets = list(self.world.targets())
        self.phase_b = FinalApproach(K, self.targets)
        self.rows: list[dict] = []
        self.ticks = 1
        self.tag = "drive"
        self.decision = None
        self.measure("start", 0)

    # -- truth + measurement for the current frame

    def truth(self) -> dict:
        x, y, yaw = self.world.pose()
        dist, heading = self.world.distance_heading()
        return {"x": x, "y": y, "yaw_deg": math.degrees(yaw), "dist_m": dist, "heading_deg": heading}

    def measure(self, action: str, ticks: int) -> dict:
        move = self.phase_b.step(self.frame)
        self.decision = move
        est = self.phase_b.last
        row = {"n": len(self.rows), "tag": self.tag, "action": action, "ticks": ticks, **self.truth(),
               "decision": str(move), "note": self.phase_b.rows[-1]["note"],
               **{f"m{i}": c for i, c in enumerate(self.phase_b.matcher.counts)}}
        if est is not None:
            row.update(view=est.view, inliers=est.inliers, yaw_front=round(est.yaw_front, 2),
                       parallax=round(est.parallax, 2), bearing=round(est.bearing_right, 1),
                       sees_goal=int(self.phase_b.sees_goal(est)))
        self.rows.append(row)
        return row

    def move(self, action: str, ticks: int) -> dict:
        self.frame = self.world.step(getattr(sim, action), ticks)[-1]
        return self.measure(action, ticks)

    # -- scripted sequences

    def turn_calibration(self) -> None:
        self.tag = "turn"
        start = self.truth()["yaw_deg"]
        total, n = 0.0, 0
        last = start
        while total < 360 and n < 400:
            yaw = self.move("LEFT", 1)["yaw_deg"]
            total += abs(((yaw - last) + 180) % 360 - 180)
            last = yaw
            n += 1
            self.show()
        print(f"turn: {n} LEFT ticks for {total:.0f} deg -> {total / n:.2f} deg per tick")
        self.tag = "drive"

    def goal_sweep(self, back: int = 20) -> None:
        self.tag = "sweep"
        self.turn_calibration()
        self.tag = "sweep"
        for action, undo in (("BACKWARD", "FORWARD"), ("FORWARD", "BACKWARD")):
            start = self.truth()
            for _ in range(back):
                self.move(action, 1)
                self.show()
            end = self.truth()
            step = math.hypot(end["x"] - start["x"], end["y"] - start["y"]) / back
            print(f"{action}: {step:.3f} m per tick (short if it hit a wall)")
            for _ in range(back):
                self.move(undo, 1)
                self.show()
        print(f"sweep done, back at {self.truth()['dist_m']:.2f} m from the goal")
        self.tag = "drive"

    # -- closed loop: phase B drives, the truth judges

    def auto(self, limit: int = 150) -> tuple[str, float, float]:
        """Let phase B drive from here until it checks in, gives up, or runs out of moves."""
        tag, self.tag = self.tag, "auto"
        self.phase_b.reset_budget()
        self.phase_b._cooldown = 0
        self.measure("auto-start", 0)
        outcome = "out of moves"
        ticks = 0
        for _ in range(limit):
            move = self.decision
            if move is None:
                outcome = "no move: " + self.phase_b.rows[-1]["note"]
                break
            if move == Action.CHECKIN:
                outcome = "CHECKIN"
                break
            action, n = move
            ticks += n
            self.phase_b.charge(move)
            self.move(action.name, n)
            self.show()
        t = self.truth()
        print(f"auto: {outcome} at {t['dist_m']:.3f} m, heading off {t['heading_deg']:.1f} deg, {ticks} ticks")
        self.tag = tag
        return outcome, t["dist_m"], t["heading_deg"], ticks

    def trials(self, n: int = 20) -> None:
        """Scramble the pose near the goal, let phase B recover, repeat. Start this on the goal."""
        rng = np.random.default_rng(len(self.rows))
        results = []
        for k in range(n):
            self.tag = "scramble"
            for action, ticks in ((rng.choice(["LEFT", "RIGHT"]), int(rng.integers(0, 50))),
                                  (rng.choice(["FORWARD", "BACKWARD"]), int(rng.integers(0, 9))),
                                  (rng.choice(["LEFT", "RIGHT"]), int(rng.integers(0, 50)))):
                if ticks:
                    self.move(str(action), ticks)
            start = self.truth()
            self.show()
            if start["dist_m"] > 0.25:
                print(f"stopping the trials: this scramble started {start['dist_m']:.2f} m out, beyond phase B's "
                      "range (~0.3 m). Drive back to the goal and press r again.")
                break
            outcome, dist, heading, ticks = self.auto()
            results.append((start["dist_m"], start["heading_deg"], outcome, dist, heading, ticks))
        n = len(results)
        if not n:
            self.tag = "drive"
            return
        print(f"\n{'start m':>8s} {'start deg':>9s}  {'end m':>6s} {'end deg':>7s} {'ticks':>5s}  outcome")
        for sd, sh, o, d, h, t in results:
            print(f"{sd:8.2f} {sh:9.1f}  {d:6.3f} {h:7.1f} {t:5d}  {o}")
        ok = [r for r in results if r[2] == "CHECKIN"]
        full = [r for r in ok if r[3] <= 0.1 and r[4] <= 10]
        if ok:
            ideal = [r[1] / self.phase_b.p.turn_deg_per_tick for r in results]  # just turning to face: a lower bound
            print(f"checked in {len(ok)}/{n}, full distance+heading points {len(full)}/{n}; worst "
                  f"{max(r[3] for r in ok):.3f} m, {max(r[4] for r in ok):.1f} deg; ticks median "
                  f"{int(np.median([r[5] for r in results]))} (turning alone would need ~{int(np.median(ideal))})")
        self.tag = "drive"

    # -- window (tkinter: the project's OpenCV is a build without windows)

    def show(self) -> None:
        r = self.rows[-1]
        view = np.hstack([self.frame, self.targets[0]])
        view = cv2.resize(view, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        ok, png = cv2.imencode(".png", view)
        self._photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))  # keep a reference
        self._image.configure(image=self._photo)
        self._text.configure(text="\n".join([
            f"TRUE: {r['dist_m']:.2f} m, heading off {r['heading_deg']:.1f} deg   ticks/key {self.ticks}   [{self.tag}]",
            f"phase B: {self.phase_b.status}"[:140],
            "arrows/WASD move  1-9 ticks/key  t turn-cal  g goal sweep  p phase B drives  r 20 trials  q quit",
        ]))
        self._root.update()

    def on_key(self, event) -> None:
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self._root.quit()
            return
        if key in KEYS:
            self.phase_b._expect = None  # your move, not phase B's probe: don't judge it
            self.move(KEYS[key], self.ticks)
        elif key.isdigit() and key != "0":
            self.ticks = int(key)
        elif key == "t":
            self.turn_calibration()
        elif key == "g":
            self.goal_sweep()
        elif key == "p":
            self.auto()
        elif key == "r":
            self.trials()
        self.show()

    def run(self) -> None:
        self._root = tk.Tk()
        self._root.title("calibrate  (left: camera, right: goal view 0)")
        self._image = tk.Label(self._root)
        self._image.pack()
        self._text = tk.Label(self._root, justify="left", anchor="w", font=("Consolas", 11))
        self._text.pack(fill="x")
        self._root.bind("<Key>", self.on_key)
        self._root.protocol("WM_DELETE_WINDOW", self._root.quit)
        self.show()
        self._root.focus_force()
        self._root.mainloop()
        self._root.destroy()

    # -- output

    def save(self) -> Path:
        path = Path("logs") / f"calibrate_{self.maze}.csv"
        path.parent.mkdir(exist_ok=True)
        fields = list(dict.fromkeys(k for r in self.rows for k in r))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.rows)
        return path


def summary(rows: list[dict], p) -> None:
    front = [r for r in rows if r.get("view") == 0]
    print(f"\n{len(rows)} frames, {len(front)} with the front view as best match")
    print("front-view inliers (median / max, frame count) by TRUE distance and heading:")
    dbins = [0, 0.1, 0.2, 0.4, 0.8, 99]
    hbins = [0, 10, 30, 181]
    print("  dist \\ heading " + "".join(f"{f'{a}-{b if b < 181 else 180}':>18s}" for a, b in zip(hbins, hbins[1:])))
    for d0, d1 in zip(dbins, dbins[1:]):
        cells = []
        for h0, h1 in zip(hbins, hbins[1:]):
            v = [r["inliers"] for r in front if d0 <= r["dist_m"] < d1 and h0 <= abs(r["heading_deg"]) < h1]
            cells.append(f"{int(np.median(v)):>3d} / {max(v):>3d} ({len(v):>3d})" if v else f"{'-':>18s}")
        print(f"  {d0:.1f}-{d1 if d1 < 99 else 'inf':<5}    " + "".join(f"{c:>18s}" for c in cells))
    yaw = [(r["yaw_front"], r["heading_deg"]) for r in front if r["dist_m"] < 0.3 and r["inliers"] >= p.min_inliers]
    if yaw:
        err = [abs(abs(a) - abs(b)) for a, b in yaw]
        print(f"|yaw_front| vs true heading within 0.3 m: median error {np.median(err):.1f} deg over {len(err)} frames")
    checkin = [r for r in rows if str(r.get("note", "")).startswith("check in")]
    if checkin:
        good = sum(r["dist_m"] <= 0.2 and abs(r["heading_deg"]) <= 10 for r in checkin)
        print(f"phase B would check in on {len(checkin)} frames; {good} of them within 0.2 m and 10 deg; "
              f"worst: {max(r['dist_m'] for r in checkin):.2f} m, {max(abs(r['heading_deg']) for r in checkin):.1f} deg")
    else:
        print("phase B would not have checked in on any frame")
    engage = [r for r in rows if r.get("sees_goal")]
    if engage:
        print(f"phase B would take over on {len(engage)} frames, up to {max(r['dist_m'] for r in engage):.2f} m from the goal")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0, help="motion noise seed")
    args = ap.parse_args()
    cal = Calibrator(args.maze, args.seed)
    print(f"maze {args.maze}: {len(cal.targets)} goal views; start {cal.truth()['dist_m']:.2f} m from the goal")
    try:
        cal.run()
    finally:
        path = cal.save()
        print(f"wrote {path}")
        summary(cal.rows, cal.phase_b.p)


if __name__ == "__main__":
    main()
