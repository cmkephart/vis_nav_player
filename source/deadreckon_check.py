"""
Can we draw a usable map of the maze from the exploration drives' recorded actions alone (dead reckoning)?
Local only: re-records a seed's exploration drives in the simulator while logging the TRUE pose after every
step, then compares that with integrating the actions at a fixed rate per tick. No GUI, writes nothing in data/.

    uv run source/deadreckon_check.py 21
    uv run source/deadreckon_check.py 7 --plot      # also logs/deadreckon_<seed>.png (truth vs dead reckoning)

Answers: (1) per-tick turn / drive and their noise, (2) how far dead reckoning drifts per drive, raw and with
headings snapped to multiples of 90 deg, (3) whether the corridors are axis-aligned (yaw while driving vs 90 deg),
(4) what the simulator object exposes (in case it offers the maze layout or the shortest route).
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import vis_nav_sim as sim


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def name_of(action) -> str:
    for n in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
        if action == getattr(sim, n, object()):
            return n
    s = str(getattr(action, "name", action)).upper()
    return "+".join(n for n in ("FORWARD", "BACKWARD", "LEFT", "RIGHT") if n in s) or s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    world = sim.Simulator(sim.Textures.find(), args.seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=args.seed)
    print("simulator exposes:", [a for a in dir(world) if not a.startswith("_")])
    log: list[tuple[str, int, tuple, tuple]] = []
    inner = world.step

    def step(action, ticks=1, *a, **k):
        before = tuple(world.pose())
        out = inner(action, ticks, *a, **k)
        log.append((name_of(action), int(ticks), before, tuple(world.pose())))
        return out

    world.step = step
    with tempfile.TemporaryDirectory() as tmp:
        summary = world.record(Path(tmp) / "rec", routes=3, seed=args.seed, capture_every=5)
    print("record summary:", summary)
    if not log:
        print("\nrecord() does not go through world.step, so the true poses were not logged.\n"
              "Paste this output to Claude: the list above shows what else the simulator offers.")
        return

    # split into drives: the robot is put back at the spawn between drives (a pose jump between two steps)
    drives, cur = [], []
    for name, ticks, p0, p1 in log:
        if cur:
            q = cur[-1][3]
            if math.hypot(p0[0] - q[0], p0[1] - q[1]) > 1e-6 or abs(wrap(p0[2] - q[2])) > 1e-6:
                drives.append(cur); cur = []
        cur.append((name, ticks, p0, p1))
    if cur:
        drives.append(cur)
    print(f"{len(log)} step calls, {sum(e[1] for e in log)} ticks, {len(drives)} drives, "
          f"spawns: {[tuple(round(v, 2) for v in d[0][2]) for d in drives]}")

    # per-tick motion and its noise
    turn, drive = {"LEFT": [], "RIGHT": []}, {"FORWARD": [], "BACKWARD": []}
    for d in drives:
        for name, ticks, p0, p1 in d:
            if name in turn:
                turn[name].append(math.degrees(wrap(p1[2] - p0[2])) / ticks)
            elif name in drive:
                drive[name].append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / ticks)
    for k, v in {**turn, **drive}.items():
        if v:
            print(f"{k:8s}: {np.mean(v):+.4f} per tick (std {np.std(v):.4f}, n={len(v)})  "
                  + ("deg" if k in turn else "m"))
    rate_l = np.mean(turn["LEFT"]) if turn["LEFT"] else 3.44
    rate_r = np.mean(turn["RIGHT"]) if turn["RIGHT"] else -3.44
    rate_f = np.mean(drive["FORWARD"]) if drive["FORWARD"] else 0.029
    rate_b = np.mean(drive["BACKWARD"]) if drive["BACKWARD"] else 0.029

    # how far the true heading is from a multiple of 90 deg while driving straight
    off = [abs(((math.degrees(p0[2]) + 45) % 90) - 45) for d in drives for n, _, p0, _ in d if n == "FORWARD"]
    if off:
        print(f"yaw while driving, distance to a multiple of 90 deg: median {np.median(off):.1f}, "
              f"90% under {np.percentile(off, 90):.1f} deg  (small = axis-aligned corridors)")

    # dead reckoning per drive: raw, and with the driving direction snapped to 90 deg
    paths = []
    for i, d in enumerate(drives):
        x = y = 0.0
        yaw = d[0][2][2]
        sx, sy = x, y
        tx0, ty0 = d[0][2][0], d[0][2][1]
        err, err_s, track = [], [], []
        for name, ticks, p0, p1 in d:
            for part in name.split("+"):
                if part == "LEFT":
                    yaw += math.radians(rate_l) * ticks
                elif part == "RIGHT":
                    yaw += math.radians(rate_r) * ticks
            dist = (rate_f if "FORWARD" in name else -rate_b if "BACKWARD" in name else 0.0) * ticks
            snap = round(yaw / (math.pi / 2)) * (math.pi / 2)
            x += dist * math.cos(yaw); y += dist * math.sin(yaw)
            sx += dist * math.cos(snap); sy += dist * math.sin(snap)
            tx, ty = p1[0] - tx0, p1[1] - ty0
            err.append(math.hypot(x - tx, y - ty)); err_s.append(math.hypot(sx - tx, sy - ty))
            track.append((tx, ty, x, y, sx, sy))
        heading_err = math.degrees(wrap(yaw - d[-1][3][2]))
        ticks = sum(t for _, t, _, _ in d)
        print(f"drive {i}: {ticks} ticks, true path {sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(track, track[1:])):.1f} m | "
              f"position error raw: median {np.median(err):.2f} m, max {max(err):.2f} m | snapped: median "
              f"{np.median(err_s):.2f}, max {max(err_s):.2f} m | heading error at the end {heading_err:+.1f} deg")
        paths.append(np.array(track))

    if args.plot:
        import cv2
        allp = np.vstack(paths)
        lo, hi = allp.min(0)[:2].min(), allp.max(0)[:2].max()
        lo = min(lo, allp[:, 2:].min()); hi = max(hi, allp[:, 2:].max())
        size, pad = 800, 20
        px = lambda x, y: (int(pad + (x - lo) / (hi - lo + 1e-9) * (size - 2 * pad)),  # noqa: E731
                           int(size - pad - (y - lo) / (hi - lo + 1e-9) * (size - 2 * pad)))
        img = np.full((size, size, 3), 255, np.uint8)
        for p in paths:
            for (a, b), colour in (((0, 1), (0, 0, 0)), ((2, 3), (0, 0, 255)), ((4, 5), (0, 160, 0))):
                pts = np.array([px(r[a], r[b]) for r in p], np.int32)
                cv2.polylines(img, [pts], False, colour, 1)
        cv2.putText(img, "black: truth   red: dead reckoning   green: snapped to 90 deg", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        Path("logs").mkdir(exist_ok=True)
        cv2.imwrite(f"logs/deadreckon_{args.seed}.png", img)
        print(f"wrote logs/deadreckon_{args.seed}.png")


if __name__ == "__main__":
    main()
