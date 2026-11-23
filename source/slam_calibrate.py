"""
Learn the robot's motion model (how far one recorded frame moves / turns) from the LOCAL simulator.

The simulator has momentum: a frame's motion depends on the moves before it (a turn keeps rotating through the next IDLE
frames, a RIGHT right after FORWARD still slides forward). So the model is a table keyed on the last 3 actions,
with fallbacks to the last 2 and the last 1. The server runs the same physics, so this is a one-off calibration of the
robot, not knowledge of any maze. Truth poses come only from local recordings (poses.json), never from the server.

    uv run python source/slam_calibrate.py 1 2 3 4 5        # seeds to record and fit on (writes source/motion_model.json)
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("motion_model.json")
CONTEXT = 3


def spawn_of(seed: int):
    import vis_nav_sim as sim
    return sim.Simulator(sim.Textures.find(), seed).spawn


def record_truth(seed: int, root: Path) -> Path:
    import vis_nav_sim as sim
    path = root / f"truth-{seed}"
    if not (path / "traj_0" / "poses.json").exists():
        world = sim.Simulator(sim.Textures.find(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
        world.record(path, routes=3, seed=seed, capture_every=5, student_files=False)
    return path


def samples(path: Path, spawn):
    """(context actions, forward m, left m, turn rad CCW) for every recorded frame, in the frame of the previous pose
    (the first frame of a drive moves from the spawn, where the robot stands still)."""
    for d in sorted(path.glob("traj_*")):
        acts = [r["action"][0] if len(r["action"]) == 1 else "IDLE" for r in json.loads((d / "data_info.json").read_text())]
        p = np.array([list(spawn)] + json.loads((d / "poses.json").read_text()))
        acts = ["START"] + acts
        for i in range(1, len(p)):
            x0, y0, t0 = p[i - 1]
            x1, y1, t1 = p[i]
            c, s = math.cos(t0), math.sin(t0)
            ctx = tuple(acts[j] if j >= 1 else "START" for j in range(i - CONTEXT + 1, i + 1))
            yield ctx, (x1 - x0) * c + (y1 - y0) * s, -(x1 - x0) * s + (y1 - y0) * c, \
                (t1 - t0 + math.pi) % (2 * math.pi) - math.pi


def fit(rows, min_n=12):
    table = {}
    for L in range(1, CONTEXT + 1):
        g = defaultdict(list)
        for ctx, f, l, t in rows:
            g[ctx[-L:]].append((f, l, t))
        for k, v in g.items():
            if len(v) >= min_n:
                v = np.array(v)
                table["|".join(k)] = dict(n=len(v), f=float(v[:, 0].mean()), l=float(v[:, 1].mean()), t=float(v[:, 2].mean()),
                                          sf=float(max(v[:, 0].std(), 0.005)), sl=float(max(v[:, 1].std(), 0.005)),
                                          st=float(max(v[:, 2].std(), math.radians(0.5))))
    return table


def vo_scale(paths, pairs_per_seed=500):
    """Visual-odometry rotation scale: true rotation / essential-matrix rotation over consecutive frames (fit on inliers)."""
    import cv2
    import slam_map as M
    import tracking_agent as T
    Y, Tt = [], []
    for path in paths:
        d = path / "traj_0"
        recs = json.loads((d / "data_info.json").read_text())
        p = np.array(json.loads((d / "poses.json").read_text()))
        keep = [k for k, r in enumerate(recs) if r["action"][0] != "IDLE"][:pairs_per_seed + 1]
        f = [T.rootsift(cv2.imread(str(d / recs[k]["image"]))) for k in keep]
        for (a, fa), (b, fb) in zip(zip(keep, f), zip(keep[1:], f[1:])):
            A, B = T._inlier_points(fa, fb)
            y = M.rel_yaw(A, B)
            if y is not None:
                Y.append(y); Tt.append((p[b, 2] - p[a, 2] + math.pi) % (2 * math.pi) - math.pi)
    Y, Tt = np.array(Y), np.array(Tt)
    ok = np.abs(Y - Tt) < math.radians(5)
    return float(np.sum(Y[ok] * Tt[ok]) / np.sum(Y[ok] ** 2)), int(ok.sum())


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5]
    root = Path("cache") / "slam_truth"
    rows = []
    for s in seeds:
        print(f"seed {s}: recording / loading truth poses ...", flush=True)
        rows += list(samples(record_truth(s, root), spawn_of(s)))
    table = fit(rows)
    vs, nvs = vo_scale([root / f"truth-{s}" for s in seeds[:3]])
    print(f"visual-odometry rotation scale {vs:.4f} (from {nvs} frame pairs)")
    OUT.write_text(json.dumps(dict(context=CONTEXT, seeds=seeds, vo_scale=vs, table=table), indent=1))
    print(f"{len(rows)} frames from seeds {seeds}: {len(table)} contexts -> {OUT}")
