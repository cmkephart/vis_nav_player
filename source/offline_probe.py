"""
Offline probe: no simulator, no GPU. Matches every exploration frame against the goal photo(s)
in the dataset folder and lists the frames that match best -- i.e. where the goal can be seen
from, and whether the baseline's goal node is anywhere near it.

    uv run source/offline_probe.py                       # data/local-7
    uv run source/offline_probe.py data/local-7 --top 20
    uv run source/offline_probe.py data/local-21 --views logs --show 2069 2302 2305 1420

--views DIR reads the goal views a session saved (DIR/goal_view_0..3.png) instead of target*.
Node numbers follow vlad.load_frames with --subsample (default 1, as route_agent/tracking_agent),
so they match the goal vote's clusters. --show N... writes logs/compare_node<N>.png: that node's
frame beside the goal view it matched, with its inliers. The top match is saved as logs/compare_best.png.

Writes logs/offline_probe.csv. Assumes the layout vlad.load_frames reads (traj_*/data_info.json
with an "image" per record) and goal photos named target*.jpg/png in the dataset folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from final_approach import GoalMatcher, Params

MOVEMENT = {"FORWARD", "BACKWARD", "LEFT", "RIGHT"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default="data/local-7")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--node", type=int, help="also list the frames around this baseline node (e.g. its goal node)")
    ap.add_argument("--around", type=int, default=15, help="how many nodes either side for --node")
    ap.add_argument("--f", type=float, default=92.0, help="focal length in px (from your K)")
    ap.add_argument("--views", help="folder with goal_view_0..3.png saved by a session (e.g. logs)")
    ap.add_argument("--subsample", type=int, default=1, help="node numbering, as the agent's --subsample")
    ap.add_argument("--show", type=int, nargs="*", default=[], help="nodes to save beside goal view 0")
    args = ap.parse_args()

    data = Path(args.data)
    images = []
    goals = []
    if args.views:
        vp = [Path(args.views) / f"goal_view_{k}.png" for k in range(4)]
        missing = [str(p) for p in vp if not p.exists()]
        if missing:
            raise SystemExit(f"missing goal views: {missing}")
        images = [cv2.imread(str(p)) for p in vp]
        print(f"goal views from {args.views}/goal_view_0..3.png (front, left, back, right)")
    else:
        goals = sorted(p for p in data.glob("target*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not goals:
            raise SystemExit(f"no target*.jpg/png in {data}: {sorted(p.name for p in data.iterdir())[:20]}"
                             "\n(use --views logs to read the goal views the last session saved)")
    for p in goals:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        n = max(1, round(w / h / (4 / 3)))      # a 1280x240 strip is four 320x240 views side by side
        tw = w // n
        images += [img[:, i * tw:(i + 1) * tw] for i in range(n)]
        print(f"{p.name}: {w}x{h} -> {n} view(s) of {tw}x{h}" + ("  (front, left, back, right)" if n == 4 else ""))
    h, w = images[0].shape[:2]
    K = np.array([[args.f, 0, w / 2], [0, args.f, h / 2], [0, 0, 1]])
    print(f"f={args.f}, principal point ({w / 2:.0f}, {h / 2:.0f})")
    matcher = GoalMatcher(K, images, Params())

    trajs = sorted(
        (d for d in data.iterdir() if d.is_dir() and d.name.startswith("traj_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    # which baseline node each frame is (or is nearest to), mirroring vlad.load_frames
    kept: dict[str, list[tuple[int, int]]] = {}
    kept_idx: dict[str, set[int]] = {}
    total = 0
    for traj in trajs:
        records = json.loads((traj / "data_info.json").read_text())
        idx = [i for i, r in enumerate(records) if len(r["action"]) == 1 and r["action"][0] in MOVEMENT][::args.subsample]
        kept[traj.name] = [(i, total + k) for k, i in enumerate(idx)]
        kept_idx[traj.name] = set(idx)
        total += len(idx)
    print(f"nodes: {total} (subsample {args.subsample}; numbering as vlad.load_frames)")

    def node_near(name: str, i: int):
        pairs = kept[name]
        return min(pairs, key=lambda p: abs(p[0] - i))[1] if pairs else ""

    rows = []
    for traj in sorted((d for d in data.iterdir() if d.is_dir() and d.name.startswith("traj_")),
                       key=lambda d: int(d.name.split("_")[1])):
        records = json.loads((traj / "data_info.json").read_text())
        for i, r in enumerate(records):
            img = cv2.imread(str(traj / r["image"]))
            if img is None:
                continue
            est = matcher.estimate(img)
            rows.append({
                "traj": traj.name, "i": i, "node": node_near(traj.name, i),
                "is_node": int(i in kept_idx[traj.name]), "image": r["image"], "action": "".join(r["action"]) if isinstance(r["action"], list) else r["action"],
                "best_raw": max(matcher.counts), **{f"m{k}": c for k, c in enumerate(matcher.counts)}, "inliers": est.inliers if est else 0,
                "view": est.view if est else "", "yaw_front": round(est.yaw_front, 1) if est else "",
                "parallax": round(est.parallax, 1) if est else "",
            })
    rows.sort(key=lambda r: (r["inliers"], r["best_raw"]), reverse=True)
    Path("logs").mkdir(exist_ok=True)
    with Path("logs/offline_probe.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

    if args.node is not None:
        near = sorted((r for r in rows if r["is_node"] and abs(r["node"] - args.node) <= args.around), key=lambda r: r["node"])
        print(f"\nframes around node {args.node} (+-{args.around}); raw matches per view: front,left,back,right")
        for r in near:
            per = ",".join(str(r.get(f"m{k}", "")) for k in range(len(images)))
            mark = "  <== node " + str(args.node) if r["node"] == args.node else ""
            print(f"  node {r['node']:4d}  {r['traj']}/{r['image']:<10s} raw {per:>14s}  inliers {r['inliers']:3d}{mark}")
        hit = [r for r in near if r["node"] == args.node]
        if hit:
            print(f"open this frame and compare it with the goal photo: {data / hit[0]['traj'] / hit[0]['image']}")
        print()
    n = len(rows)
    print(f"{n} frames; with >=15 inliers: {sum(r['inliers'] >= 15 for r in rows)}, "
          f">=30: {sum(r['inliers'] >= 30 for r in rows)}")
    print(f"{'traj':8s} {'i':>5s} {'node':>5s} {'raw':>4s} {'inl':>4s} view   yaw   par  per-view raw  image")
    for r in rows[: args.top]:
        per = ",".join(str(r.get(f"m{k}", "")) for k in range(len(images)))
        print(f"{r['traj']:8s} {r['i']:5d} {str(r['node']):>5s} {r['best_raw']:4d} {r['inliers']:4d} {str(r['view']):>4s} {str(r['yaw_front']):>5s} {str(r['parallax']):>5s}  {per:>12s}  {r['image']}")

    def save_pair(r, name):
        live = cv2.imread(str(data / r["traj"] / r["image"]))
        v = r["view"] if r["view"] != "" else 0
        goal = images[int(v)]
        if live is None:
            return
        live = cv2.resize(live, (goal.shape[1], goal.shape[0]))
        pair = np.hstack([live, goal])
        cv2.putText(pair, f"node {r['node']} {r['traj']}/{r['image']}  inliers {r['inliers']} view {r['view']}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(pair, f"goal view {v} ({['front', 'left', 'back', 'right'][int(v)] if len(images) == 4 else v})", (goal.shape[1] + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imwrite(str(Path("logs") / name), pair)

    if rows:
        save_pair(rows[0], "compare_best.png")
    by_node = {r["node"]: r for r in rows if r["is_node"]}
    if args.show:
        print("\nrequested nodes (best of the node's own frame):")
    for nd in args.show:
        r = by_node.get(nd)
        if r is None:
            print(f"  node {nd}: not found (have 0..{total - 1})")
            continue
        save_pair(r, f"compare_node{nd}.png")
        near = [x for x in rows if x["is_node"] and abs(x["node"] - nd) <= 3]
        best = max(near, key=lambda x: x["inliers"])
        print(f"  node {nd:5d}: inliers {r['inliers']:3d} (best within +-3: {best['inliers']} at node {best['node']})"
              f"  -> logs/compare_node{nd}.png")
    print("saved logs/compare_best.png (top match beside the goal view it matched)")


if __name__ == "__main__":
    main()
