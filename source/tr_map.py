"""
Teach-and-repeat map: everything phase A needs, computed once from the exploration data and cached.

Why this design (see the notes in nav_agent.py): the exploration drives are the only routes we KNOW the robot can
drive -- seeing through a gap in a wall never creates a link here. A route is a path along recorded drives,
joined only where two frames were taken from the SAME position (a pure rotation explains the matches), so
switching between them is a turn on the spot. Live tracking then follows that one route, not the whole maze.

    uv run source/tr_map.py data/local-21            # build + report (offline, no simulator)

Cached under cache/<data folder name>/tr_*.pkl.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

K_DEFAULT = np.array([[92.0, 0, 160], [0, 92.0, 120], [0, 0, 1]])
TICKS_PER_RECORD = 5
DEG_PER_TICK = 3.44
N_FEATURES = 800
RATIO = 0.8
# same-pose links
THUMB = (32, 24)
THUMB_SHIFT = 6               # +-px at thumbnail size (~+-23 deg)
THUMB_TOP = 8
THUMB_MIN = 0.6
MIN_GAP = 15                  # records apart within one drive
LINK_PARALLAX = 6.0           # px left after removing the best pure rotation
LINK_INLIERS = 30
LINK_BINS = 5                 # matches must fill >= this many of 8 vertical strips
# goal candidates
GOAL_WINDOW = 3
GOAL_BINS = 2                 # layout spread cannot tell the goal from a look-alike (maze 21: 4 vs 3-4 strips);
                              # the person picks among the candidates, so only drop single-patch matches
GOAL_SEPARATION = 20          # records between two candidates of the same drive
VIEW_NAMES = ["FRONT", "LEFT", "BACK", "RIGHT"]


def rootsift(gray: np.ndarray):
    sift = rootsift.sift = getattr(rootsift, "sift", None) or cv2.SIFT_create(nfeatures=N_FEATURES)
    kp, d = sift.detectAndCompute(gray, None)
    if d is None or len(kp) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7)).astype(np.float32)
    return np.float32([k.pt for k in kp]), d


_BF = cv2.BFMatcher()


def ratio_pairs(f1, f2, ratio: float = RATIO):
    (p1, d1), (p2, d2) = f1, f2
    if len(d1) < 8 or len(d2) < 8:
        return None, None
    good = [m for m, n in (x for x in _BF.knnMatch(d1, d2, k=2) if len(x) == 2) if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, None
    return (p1[[m.queryIdx for m in good]].astype(np.float64), p2[[m.trainIdx for m in good]].astype(np.float64))


def strips(xs: np.ndarray, width: float = 320.0, bins: int = 8, per: int = 3) -> int:
    return int((np.histogram(xs, bins=bins, range=(0.0, width))[0] >= per).sum())


def rotation_fit(a: np.ndarray, b: np.ndarray, K: np.ndarray, px: float = 3.0):
    """Best pure rotation a -> b. Returns (yaw_left_deg, parallax_px, inliers, strips) or None.
    yaw_left > 0: camera b looks further LEFT than camera a."""
    if a is None or len(a) < 12:
        return None
    H, mask = cv2.findHomography(a, b, cv2.RANSAC, px)
    if H is None:
        return None
    keep = mask.ravel() > 0
    if keep.sum() < 10:
        return None
    U, _, Vt = np.linalg.svd(np.linalg.inv(K) @ H @ K)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    moved = cv2.perspectiveTransform(a[keep].reshape(-1, 1, 2), K @ R @ np.linalg.inv(K)).reshape(-1, 2)
    parallax = float(np.median(np.linalg.norm(moved - b[keep], axis=1)))
    yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    return yaw, parallax, int(keep.sum()), strips(a[keep][:, 0])


@dataclass
class Record:
    drive: int
    index: int          # record index inside the drive
    action: str         # "FORWARD", "LEFT", "FORWARD+LEFT", "IDLE", ...
    path: str


@dataclass
class TRMap:
    data_dir: str
    records: list[Record]
    starts: list[int]                         # global id of each drive's first record
    seq_yaw: list[float | None]               # yaw_left from record g to g+1 (same drive), None if unknown
    links: list[tuple[int, int, float, float, int]]   # (a, b, yaw_left a->b, parallax, inliers)
    goal_scores: np.ndarray | None = None     # (n, 4) best layout-checked inliers per goal view
    goal_key: str = ""
    feats: list = field(default_factory=list, repr=False)

    # ----------------------------------------------------------------- graph and routes
    def graph(self) -> nx.DiGraph:
        n = len(self.records)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for g in range(n - 1):
            if self.records[g].drive != self.records[g + 1].drive:
                continue
            G.add_edge(g, g + 1, w=TICKS_PER_RECORD, kind="drive")
            nxt = self.records[g + 1].action
            if "FORWARD" not in nxt and "BACKWARD" not in nxt:     # a turn (or a pause) can be undone on the spot
                G.add_edge(g + 1, g, w=TICKS_PER_RECORD, kind="unturn")
        for a in self.starts:                                      # every drive starts at the spawn
            for b in self.starts:
                if a != b:
                    G.add_edge(a, b, w=0.0, kind="spawn")
        for a, b, yaw, par, inl in self.links:
            w = abs(yaw) / DEG_PER_TICK + 1.0
            G.add_edge(a, b, w=w, kind="link", yaw=yaw)
            G.add_edge(b, a, w=w, kind="link", yaw=-yaw)
        return G

    def route(self, goal: int, G: nx.DiGraph | None = None) -> tuple[list[int], float]:
        G = G or self.graph()
        best = None
        for s in self.starts:
            try:
                d, p = nx.single_source_dijkstra(G, s, goal, weight="w")
            except nx.NetworkXNoPath:
                continue
            if best is None or d < best[0]:
                best = (d, p)
        if best is None:
            return [goal], math.inf
        return best[1], best[0]

    # ----------------------------------------------------------------- goal candidates
    def goal_candidates(self, k: int = 5) -> list[tuple[int, int, int, float]]:
        """Up to k (record, view, inliers, cluster score), best first. Score per VIEW (never summed across views,
        which let a far-away spot that half-sees several views win on maze 21)."""
        if self.goal_scores is None:
            return []
        n = len(self.records)
        drive = np.array([r.drive for r in self.records])
        S = self.goal_scores.astype(np.float32)
        cl = np.zeros_like(S)
        for m in range(-GOAL_WINDOW, GOAL_WINDOW + 1):
            sh = np.zeros_like(S)
            same = np.zeros(n, bool)
            if m >= 0:
                sh[: n - m] = S[m:]
                same[: n - m] = drive[: n - m] == drive[m:]
            else:
                sh[-m:] = S[: n + m]
                same[-m:] = drive[-m:] == drive[: n + m]
            cl += sh * same[:, None]
        score = cl.max(1)
        out: list[tuple[int, int, int, float]] = []
        for g in np.argsort(-score):
            if score[g] <= 0:
                break
            if any(self.records[g].drive == self.records[c].drive and abs(int(g) - c) < GOAL_SEPARATION for c, *_ in out):
                continue
            lo, hi = max(0, g - GOAL_WINDOW), min(n, g + GOAL_WINDOW + 1)
            best = max(range(lo, hi), key=lambda j: S[j].max() if drive[j] == drive[g] else -1)
            v = int(np.argmax(S[best]))
            out.append((int(best), v, int(S[best, v]), float(score[g])))
            if len(out) >= k:
                break
        return out


# --------------------------------------------------------------------- building
def load_records(data_dir: Path) -> tuple[list[Record], list[int]]:
    recs, starts = [], []
    drives = sorted((d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("traj_")),
                    key=lambda d: int(d.name.split("_")[1]))
    for di, d in enumerate(drives):
        starts.append(len(recs))
        for k, r in enumerate(json.loads((d / "data_info.json").read_text())):
            a = r["action"]
            a = "+".join(a) if isinstance(a, list) else str(a)
            recs.append(Record(di, k, a or "IDLE", str(d / r["image"])))
    return recs, starts


def _key(data_dir: Path, recs) -> str:
    h = hashlib.md5(f"{data_dir.resolve()}|{len(recs)}|{recs[-1].path}|v1".encode())
    return h.hexdigest()[:12]


def build(data_dir: Path, cache_dir: Path, K: np.ndarray = K_DEFAULT, log=print) -> TRMap:
    recs, starts = load_records(data_dir)
    n = len(recs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _key(data_dir, recs)
    fpath = cache_dir / f"tr_feats_{key}.pkl"
    mpath = cache_dir / f"tr_map_{key}.pkl"
    t0 = time.time()
    if fpath.exists():
        feats = pickle.loads(fpath.read_bytes())
    else:
        log(f"tr_map: features for {n} frames ...")
        feats = []
        for i, r in enumerate(recs):
            g = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
            feats.append(rootsift(g) if g is not None else (np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)))
            if i % 500 == 499:
                log(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)")
        fpath.write_bytes(pickle.dumps(feats))
    if mpath.exists():
        d = pickle.loads(mpath.read_bytes())
        m = TRMap(d["data_dir"], [Record(*r) for r in d["records"]], d["starts"], d["seq_yaw"], d["links"])
        m.feats = feats
        log(f"tr_map: loaded ({len(m.links)} same-pose links) in {time.time() - t0:.0f}s")
        return m

    log("tr_map: turn between consecutive frames ...")
    seq: list[float | None] = []
    for g in range(n - 1):
        if recs[g].drive != recs[g + 1].drive:
            seq.append(None)
            continue
        a, b = ratio_pairs(feats[g], feats[g + 1])
        r = rotation_fit(a, b, K) if a is not None else None
        seq.append(r[0] if r else None)
    seq.append(None)

    log(f"tr_map: same-pose links ({time.time() - t0:.0f}s) ...")
    TH = []
    for r in recs:
        g = cv2.imread(r.path, cv2.IMREAD_GRAYSCALE)
        g = np.zeros((240, 320), np.uint8) if g is None else g
        TH.append(cv2.resize(g, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32))
    TH = np.stack(TH)
    best = np.full((n, n), -1.0, np.float32)
    W = THUMB[0]
    for s in range(-THUMB_SHIFT, THUMB_SHIFT + 1):
        A = TH[:, :, max(0, s):W + min(0, s)].reshape(n, -1)
        B = TH[:, :, max(0, -s):W + min(0, -s)].reshape(n, -1)
        A = A - A.mean(1, keepdims=True)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-6
        B = B - B.mean(1, keepdims=True)
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-6
        np.maximum(best, A @ B.T, out=best)
    drive = np.array([r.drive for r in recs])
    idx = np.array([r.index for r in recs])
    best[(drive[:, None] == drive[None, :]) & (np.abs(idx[:, None] - idx[None, :]) < MIN_GAP)] = -1
    links = []
    for i in range(n):
        for j in np.argsort(-best[i])[:THUMB_TOP]:
            j = int(j)
            if j <= i or best[i, j] < THUMB_MIN:
                continue
            a, b = ratio_pairs(feats[i], feats[j])
            r = rotation_fit(a, b, K) if a is not None else None
            if r and r[1] <= LINK_PARALLAX and r[2] >= LINK_INLIERS and r[3] >= LINK_BINS:
                links.append((i, j, r[0], r[1], r[2]))
    del best
    m = TRMap(str(data_dir), recs, starts, seq, links)
    mpath.write_bytes(pickle.dumps({"data_dir": m.data_dir, "records": [(r.drive, r.index, r.action, r.path) for r in recs],
                                    "starts": starts, "seq_yaw": seq, "links": links}))
    m.feats = feats
    log(f"tr_map: built in {time.time() - t0:.0f}s: {n} frames, {len(links)} same-pose links")
    return m


def score_goal(m: TRMap, targets: list[np.ndarray], cache_dir: Path, K: np.ndarray = K_DEFAULT, log=print) -> None:
    """Fill m.goal_scores: per frame and goal view, layout-checked homography inliers (cached per goal photo set)."""
    gk = hashlib.md5(b"".join(cv2.imencode(".png", t)[1].tobytes() for t in targets)).hexdigest()[:12]
    path = cache_dir / f"tr_goal_{gk}_{len(m.records)}.npy"
    m.goal_key = gk
    if path.exists():
        m.goal_scores = np.load(path)
        return
    t0 = time.time()
    gf = [rootsift(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t) for t in targets]
    S = np.zeros((len(m.records), len(gf)), np.int32)
    for i, f in enumerate(m.feats):
        for v, g in enumerate(gf):
            a, b = ratio_pairs(g, f)
            if a is None:
                continue
            H, mask = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
            if H is None:
                continue
            keep = mask.ravel() > 0
            if keep.sum() >= 12 and strips(a[keep][:, 0]) >= GOAL_BINS:
                S[i, v] = int(keep.sum())
    np.save(path, S)
    m.goal_scores = S
    log(f"tr_map: goal views scored against {len(m.records)} frames in {time.time() - t0:.0f}s")


def split_strip(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    n = max(1, round(w / h / (4 / 3)))
    tw = w // n
    return [img[:, i * tw:(i + 1) * tw] for i in range(n)]


if __name__ == "__main__":
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data/local-21")
    cache = Path("cache") / data.name
    m = build(data, cache)
    tj = data / "target.jpg"
    if tj.exists():
        score_goal(m, split_strip(cv2.imread(str(tj))), cache)
        cands = m.goal_candidates()
        G = m.graph()
        print("goal candidates (record, drive/index, view, inliers, cluster score, route ticks):")
        for g, v, inl, sc in cands:
            r = m.records[g]
            _, d = m.route(g, G)
            print(f"  {g:5d}  traj_{r.drive}/{r.index:<5d} {VIEW_NAMES[v]:5s} {inl:4d}  {sc:6.0f}  {d:6.0f}")
