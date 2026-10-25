"""
How "rich" is each maze's front goal photo, and does that predict how many inliers phase B sees
when it stands on the goal? If it does, the check-in threshold can scale with it instead of
being a fixed 35 (which means 4 cm on maze 7 but 14 cm on maze 202).

    uv run source/goal_richness.py                 # mazes 7 3 11 25 202
    uv run source/goal_richness.py 7 3 11 25 202 58

For each maze: SIFT keypoints in goal view 0, and, if logs/calibrate_<maze>.csv exists, the
median front-view inliers phase B measured within 3 cm of the goal (heading within 3 deg).
No simulator steps are taken; it only reads the goal photos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim

import csv


def on_spot_inliers(path: Path) -> float | None:
    """Median front-view inliers within 3 cm of the goal, heading within 3 deg (no pandas needed)."""
    vals = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            try:
                if (r.get("view") in ("0", "0.0") and float(r["dist_m"]) < 0.03
                        and abs(float(r["yaw_front"])) <= 3):
                    vals.append(float(r["inliers"]))
            except (KeyError, ValueError):
                continue
    return float(np.median(vals)) if vals else None


def main() -> None:
    mazes = [int(a) for a in sys.argv[1:]] or [7, 3, 11, 25, 202]
    textures = sim.Textures.find()
    sift = cv2.SIFT_create()
    print(f"{'maze':>5s} {'keypoints':>9s} {'on-spot inliers':>16s} {'ratio':>6s}")
    rows = []
    world = None
    for m in mazes:
        if world is None:  # one simulator, reset per maze: a new one each time runs out of GPU memory
            world = sim.Simulator(textures, m, motion_noise=None, motion_seed=0)
        else:
            world.reset(seed=m, motion_seed=0)
        view0 = world.targets()[0]
        kp = sift.detect(view0, None)
        n = len(kp)
        spot = None
        log = Path("logs") / f"calibrate_{m}.csv"
        if log.exists():
            spot = on_spot_inliers(log)
        cv2.imwrite(str(Path("logs") / f"goal_view0_maze{m}.png"), view0)
        rows.append((m, n, spot))
        ratio = f"{spot / n:.3f}" if spot else "-"
        print(f"{m:5d} {n:9d} {spot if spot is not None else '-':>16} {ratio:>6s}")
    known = [(n, s) for _, n, s in rows if s]
    if len(known) >= 3:
        r = np.corrcoef(*zip(*known))[0, 1]
        print(f"\ncorrelation between keypoints and on-spot inliers: {r:+.2f} over {len(known)} mazes")
    print("goal views saved as logs/goal_view0_maze<N>.png")


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    main()
