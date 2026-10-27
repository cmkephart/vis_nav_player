"""
Package a local maze for offline development: the exploration data (what the challenge hands out) plus the
simulator's TRUTH (maze grid, spawn, goal, oracle route, and -- if it can be captured -- the true pose of every
exploration frame). Writes logs/upload_<seed>_partN.zip, each under --max-mb, to upload to Claude.

    uv run source/export_truth.py 21
    uv run source/export_truth.py 21 7          # several mazes
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import vis_nav_sim as sim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli import record_exploration_data  # noqa: E402


def jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        pass
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return repr(v)


def truth_of(seed: int) -> dict:
    world = sim.Simulator(sim.Textures.find(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
    t: dict = {"seed": seed}
    for name in ("cell_size", "spawn", "spawn_cell", "target", "target_cell", "oracle_steps", "motion_noise", "grid"):
        try:
            v = getattr(world, name)
            t[name] = jsonable(v() if callable(v) else v)
        except Exception as exc:
            t[name] = f"<error {exc}>"
    for args in ((), ("spawn",), (world.spawn_cell, world.target_cell)):
        try:
            t["shortest_path"] = jsonable(world.shortest_path(*[getattr(world, a) if isinstance(a, str) else a
                                                                for a in args]))
            t["shortest_path_args"] = repr(args)
            break
        except Exception as exc:
            t["shortest_path_error"] = repr(exc)
    for name in ("drawn_bias", "step_ticks"):
        try:
            t[name] = jsonable(getattr(world, name)())
        except Exception as exc:
            t[name] = f"<error {exc}>"

    # try to capture the true pose of every exploration step (works only if record() calls Python-level methods)
    poses: list = []
    cls = type(world)
    patched = {}
    for meth in ("step", "step_ticks", "_step", "_tick", "reset"):
        f = getattr(cls, meth, None)
        if callable(f):
            def wrap(self, *a, _f=f, _m=meth, **k):
                out = _f(self, *a, **k)
                try:
                    poses.append([_m, repr(a[:2]), list(self.pose())])
                except Exception:
                    pass
                return out
            patched[meth] = f
            setattr(cls, meth, wrap)
    try:
        w2 = sim.Simulator(sim.Textures.find(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
        with tempfile.TemporaryDirectory() as tmp:
            w2.record(Path(tmp) / "rec", routes=3, seed=seed, capture_every=5)
    except Exception as exc:
        t["record_error"] = repr(exc)
    finally:
        for meth, f in patched.items():
            setattr(cls, meth, f)
    t["record_pose_log"] = poses
    print(f"maze {seed}: truth captured; {len(poses)} pose samples from record() "
          f"({'usable' if poses else 'record() is native: no per-frame truth'})")
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--max-mb", type=float, default=24.0)
    args = ap.parse_args()
    Path("logs").mkdir(exist_ok=True)
    for seed in args.seeds:
        data = record_exploration_data(seed)
        truth = truth_of(seed)
        files = sorted(p for p in data.rglob("*") if p.is_file())
        part, size, zf, written = 0, 0.0, None, []
        limit = args.max_mb * 1024 * 1024
        for p in [None] + files:                       # None = truth.json, first in part 1
            blob_size = len(json.dumps(truth)) if p is None else p.stat().st_size
            if zf is None or size + blob_size > limit:
                if zf is not None:
                    zf.close()
                part += 1
                out = Path("logs") / f"upload_{seed}_part{part}.zip"
                zf = zipfile.ZipFile(out, "w", zipfile.ZIP_STORED)
                written.append(out)
                size = 0.0
            if p is None:
                zf.writestr(f"local-{seed}/truth.json", json.dumps(truth))
            else:
                zf.write(p, f"local-{seed}/{p.relative_to(data).as_posix()}")
            size += blob_size
        zf.close()
        print(f"maze {seed}: {len(files)} files -> " + ", ".join(f"{w} ({w.stat().st_size / 1e6:.1f} MB)" for w in written))


if __name__ == "__main__":
    main()
