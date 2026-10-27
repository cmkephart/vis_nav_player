"""
Local-only probe of the simulator (no GUI, writes only logs/). Three questions:

1. What does the simulator know about the maze? (grid, cell size, spawn/target cells, true shortest route, bias)
2. What do the recorded actions in data/local-<seed> look like? (single moves, combined moves, how many ticks)
3. If we replay those recorded actions from the spawn, does the replay reproduce the recorded drive
   (same pictures)? And how far does dead reckoning (integrating the actions) drift from the TRUE pose?

    uv run source/sim_probe.py 21
    uv run source/sim_probe.py 21 --plot      # also logs/sim_probe_<seed>.png: truth vs dead reckoning

Paste the whole output to Claude.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim

NAMES = ("FORWARD", "BACKWARD", "LEFT", "RIGHT")
TICKS_PER_RECORD = 5          # record(..., capture_every=5)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def show(label, value, limit=300):
    text = repr(value)
    print(f"  {label}: {text[:limit]}{' ...' if len(text) > limit else ''}")


def describe_world(world) -> None:
    print("== 1. what the simulator knows ==")
    for name in ("cell_size", "step_ticks", "motion_noise", "motion_bias", "drawn_bias", "spawn", "spawn_cell",
                 "target", "target_cell", "oracle_steps", "width", "height", "seed"):
        try:
            v = getattr(world, name)
            show(name, v() if callable(v) and name in ("oracle_steps",) else v)
        except Exception as exc:
            show(name, f"<error {exc}>")
    for name in ("grid", "occupancy"):
        try:
            g = getattr(world, name)
            g = g() if callable(g) else g
            arr = np.asarray(g)
            print(f"  {name}: type {type(g).__name__}, shape {arr.shape}, dtype {arr.dtype}, values {np.unique(arr)[:10]}")
        except Exception as exc:
            show(name, f"<error {exc}>")
    for name in ("shortest_path", "oracle_steps", "spl"):
        try:
            f = getattr(world, name)
            if callable(f):
                try:
                    r = f()
                    show(f"{name}()", r if not hasattr(r, "__len__") or len(r) < 40 else f"len {len(r)}: {list(r)[:15]}")
                except TypeError as exc:
                    show(f"{name}()", f"needs arguments: {exc}")
                try:
                    show(f"{name}.__doc__", (f.__doc__ or "").strip())
                except Exception:
                    pass
        except Exception as exc:
            show(name, f"<error {exc}>")
    for name in ("reset", "step", "record", "pose"):
        try:
            show(f"{name}.__doc__", (getattr(world, name).__doc__ or "").strip())
        except Exception:
            pass


def action_of(names: list[str]):
    act = None
    for n in names:
        a = getattr(sim, n)
        act = a if act is None else (act | a)
    return act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--data", default=None, help="exploration data (default data/local-<seed>)")
    args = ap.parse_args()
    data = Path(args.data or f"data/local-{args.seed}")

    world = sim.Simulator(sim.Textures.find(), args.seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=args.seed)
    describe_world(world)

    print("\n== 2. the recorded actions ==")
    trajs = sorted((d for d in data.iterdir() if d.is_dir() and d.name.startswith("traj_")),
                   key=lambda d: int(d.name.split("_")[1]))
    drives = []
    for d in trajs:
        recs = json.loads((d / "data_info.json").read_text())
        drives.append((d, recs))
        c = Counter("+".join(r["action"]) if isinstance(r["action"], list) else str(r["action"]) for r in recs)
        print(f"  {d.name}: {len(recs)} records; first keys {sorted(recs[0])}; actions {c.most_common(8)}")

    print("\n== 3. replay from the spawn ==")
    results = []
    for d, recs in drives:
        w = sim.Simulator(sim.Textures.find(), args.seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=args.seed)
        p0 = tuple(w.pose())
        x = y = 0.0
        yaw = p0[2]
        sx = sy = 0.0
        rows, diffs, bumps, bad = [], [], 0, 0
        for k, r in enumerate(recs):
            names = [a for a in (r["action"] if isinstance(r["action"], list) else [r["action"]]) if a in NAMES]
            before = tuple(w.pose())
            if names:
                try:
                    frames = w.step(action_of(names), TICKS_PER_RECORD)
                except Exception as exc:
                    bad += 1
                    if bad <= 3:
                        print(f"  step failed for {names}: {exc}")
                    continue
                img = frames[-1] if isinstance(frames, (list, tuple)) else frames
            else:
                img = None
            after = tuple(w.pose())
            # dead reckoning with the nominal rates (turn 3.44 deg, drive 0.029 m per tick)
            turn = (3.44 if "LEFT" in names else 0) - (3.44 if "RIGHT" in names else 0)
            dist = (0.029 if "FORWARD" in names else 0) - (0.029 if "BACKWARD" in names else 0)
            yaw += math.radians(turn) * TICKS_PER_RECORD
            x += dist * TICKS_PER_RECORD * math.cos(yaw); y += dist * TICKS_PER_RECORD * math.sin(yaw)
            snap = round(yaw / (math.pi / 2)) * (math.pi / 2)
            sx += dist * TICKS_PER_RECORD * math.cos(snap); sy += dist * TICKS_PER_RECORD * math.sin(snap)
            if dist and math.hypot(after[0] - before[0], after[1] - before[1]) < 0.3 * abs(dist) * TICKS_PER_RECORD:
                bumps += 1
            tx, ty = after[0] - p0[0], after[1] - p0[1]
            rows.append((tx, ty, x, y, sx, sy, wrap(yaw - after[2])))
            if img is not None and k % 50 == 0:
                rec_img = cv2.imread(str(d / r["image"]))
                try:
                    live = np.asarray(img)
                    if rec_img is not None and live.shape == rec_img.shape:
                        diffs.append(float(np.abs(live.astype(np.float32) - rec_img.astype(np.float32)).mean()))
                except Exception:
                    pass
        arr = np.array(rows)
        err = np.hypot(arr[:, 0] - arr[:, 2], arr[:, 1] - arr[:, 3])
        err_s = np.hypot(arr[:, 0] - arr[:, 4], arr[:, 1] - arr[:, 5])
        print(f"  {d.name}: {len(recs)} records, {bumps} blocked moves | replay vs recorded pictures: mean pixel diff "
              f"{np.median(diffs) if diffs else float('nan'):.1f} (0-5 = identical replay, 40+ = different path) | "
              f"dead reckoning error: median {np.median(err):.2f} m, max {err.max():.2f} m; snapped to 90: median "
              f"{np.median(err_s):.2f}, max {err_s.max():.2f} m | heading error at end {math.degrees(arr[-1, 6]):+.1f} deg")
        results.append(arr)

    if args.plot and results:
        allp = np.vstack(results)[:, :6]
        lo, hi = float(allp.min()), float(allp.max())
        size, pad = 800, 20
        px = lambda x, y: (int(pad + (x - lo) / (hi - lo + 1e-9) * (size - 2 * pad)),  # noqa: E731
                           int(size - pad - (y - lo) / (hi - lo + 1e-9) * (size - 2 * pad)))
        img = np.full((size, size, 3), 255, np.uint8)
        for arr in results:
            for (a, b), colour in (((0, 1), (0, 0, 0)), ((2, 3), (0, 0, 255)), ((4, 5), (0, 160, 0))):
                cv2.polylines(img, [np.array([px(r[a], r[b]) for r in arr], np.int32)], False, colour, 1)
        cv2.putText(img, "black: true path  red: dead reckoning  green: snapped to 90 deg", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        Path("logs").mkdir(exist_ok=True)
        out = Path("logs") / f"sim_probe_{args.seed}.png"
        cv2.imwrite(str(out), img)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
