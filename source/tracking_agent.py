r"""
Tracking agent v5.3: BaselineAgent + (1) position tracking, (2) SIFT geometric verification,
(3) verified shortcuts, (4) a route that does not replay spins/wiggles, (5) a single combined hint.

Run from the project root, exactly like the baseline:

    .\.venv\Scripts\python.exe source\tracking_agent.py --local

It subclasses BaselineAgent, so baseline_agent.py is left untouched and must sit in the same folder.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parent.parent))   # cli.py, vlad.py, baseline_agent.py, keyboard_agent.py live one folder up


from pathlib import Path

import csv
import pickle
import time
import pickle

import cli
import cv2
import networkx as nx
import numpy as np
from vis_nav_sdk import SessionInfo, run

try:
    from vis_nav_sdk import WAIT
except ImportError:                      # old stubs / SDK without it
    WAIT = "noop"

from baseline_agent import (
    MIN_SHORTCUT_GAP,
    TEMPORAL_WEIGHT,
    VISUAL_WEIGHT_BASE,
    VISUAL_WEIGHT_SCALE,
    BaselineAgent,
    _distance,
)

# ----------------------------------------------------------------------------- tunable values
WINDOW = 12               # while tracking, only look this many frames either side of the last position
ACCEPT_TRACK = 10         # SIFT matches needed to accept a nearby node while tracking
LOST_PATIENCE = 3         # keep the last position through this many weak frames before searching the whole map
STRONG_MATCH = 25         # this many matches is trusted outright (unrelated frames almost never reach it)
SECOND_RATIO = 1.3        # a weaker match must beat the best competitor elsewhere by this factor
TURN_HOP_PENALTY = 3.0    # while the user is only turning on the spot they have not moved: other nodes must win by this much per step
SHORT_RUN = 3             # a route move repeated fewer times than this does not need the view lined up with the node first
SHORT_RUN_MAX_ALIGN = 4.0 # ...unless the view is this many turn-steps (or more) off
HOP_PENALTY = 0.3         # nodes further from the last position need this many extra matches per step away
ACCEPT_GLOBAL = 20        # SIFT matches needed to accept a node after a map-wide search
GLOBAL_TOPK = 30          # how many VLAD candidates to verify with SIFT when searching the whole map
RATIO = 0.8               # Lowe ratio test for SIFT matches (lower = stricter)
SHORTCUT_MIN_INLIERS = 25 # a shortcut is only kept if SIFT confirms both frames are the same place
MIN_SPREAD = 0.2          # a link's matches must span at least this fraction of the image width (one repeated
                          # poster on a wall is not proof of the same place -- the maze reuses its textures)
LINK_SPREAD_STEPS = 8     # a verified shortcut is also tried on the nodes this many steps either side of both ends
LINK_EXTRA_MIN_INLIERS = 20  # ...and kept for each pair that still matches this well
SWITCH_MARGIN_MIN = 4     # a different node must beat the current one by at least this many matches...
SWITCH_MARGIN_FRAC = 0.12 # ...or this fraction of the current node's matches, whichever is larger
JUMP_CONFIRM_RADIUS = 6   # a far-away relocalisation must be seen twice in a row, this close together, before it is believed
HINT_HOLD = 2             # a new kind of hint must stay the same for this many updates before it replaces the shown one
DEFAULT_PX_PER_PRESS = 12.0   # until measured: how far one LEFT/RIGHT key press shifts the picture (px). Learned live, see below.
DEFAULT_NODES_PER_PRESS = 0.3 # until measured: how many recorded steps one FORWARD key press covers. Learned live.
CAL_PRIOR = 8                 # how many "pretend" samples the defaults count for before real measurements take over
DEEP_KEY = "h"                # press this to run a slow, thorough position check (set to None to disable)
DEEP_TOPK = 400               # the deep check verifies this many look-alike candidates, not just the usual 30
NOTICE_TICKS = 30             # how long the deep-check result stays on screen
DEPTH_CLIP = 4.0          # never correct the position by more than this many steps from the depth cue
SCALE_SIGMA = 0.6         # a node whose keypoints are this much (log2) bigger/smaller than yours counts for only ~60%
SCALE_PER_STEP = 0.35     # measured: one FORWARD step grows matched keypoints by about this much (log2)
HINT_STYLE = "decision"   # "decision": quiet on straight stretches, speak up at turns/junctions; "always": old behaviour
QUIET_TURN = 2.0          # a route that turns by less than this (turn-steps) over the next few moves counts as 'straight on'
QUIET_ALIGN = 3.0         # on a straight stretch, ignore view misalignment smaller than this many turn-steps
DECISION_MOVES = 6        # how many real moves ahead to look for the next turn or junction
STEER_ENABLED = True      # steer from the live view itself: compare it with frames further along the route
STEER_TARGETS = (24, 16, 10, 6, 3, 1)  # try the farthest of these route frames first, then come closer until one is in view
STEER_LOOKAHEAD = 30      # never look further than this many route nodes
STEER_TURN_PX = 30        # picture offset (px) that makes it say "turn" (about the noise level of a straight walk is 14 px)
STEER_KEEP_PX = 15        # ...and it keeps saying turn the same way until the offset is below this
STEER_DEPTH_MIN = 0.10    # |log2 size ratio| below this = the target is at your distance; above = it is ahead (+) or behind (-)
STEER_WINDOW = 8          # forward/backward is decided from the NET movement over this many real moves of the route
STEER_REACH = 2           # the heading is measured against a frame at most this many real moves ahead (further ahead,
                          # walking itself shifts the picture sideways and would be mistaken for a heading error)
STEER_MIN_INLIERS = 18    # the live view must share this many matches with that frame for it to be used as the target
DEG_PER_PX = 0.28         # rough conversion for the on-screen degrees (assumes a ~90 degree camera over 320 px)
BLOCKED_DIFF = 3.5        # mean pixel change below this after a FORWARD/BACKWARD key means the move did nothing
BLOCKED_COUNT = 2         # ...this many times in a row = you are pushing against a wall
SHORTCUT_SUPPRESS = 15    # don't pick two shortcuts whose ends are both within this many nodes
SHORTCUT_POOL = 60        # look at top_k * this many VLAD pairs when choosing shortcuts
LOOKAHEAD_MAX = 40        # the hint looks at most this many route steps ahead for the next real move
BACKWARD_COST = 3.0       # cost of walking BACKWARD along the recording (forward/turn steps cost 1.0)
SAME_POSE_SPAN = 60       # look for "same view again" links between frames up to this many steps apart
SAME_POSE_MIN_INLIERS = 50  # SIFT matches needed to call two frames the same pose
SAME_POSE_MAX_SHIFT = 15  # ...and their views may differ by at most this many pixels
SAME_POSE_WEIGHT = 0.3    # cost of hopping between two frames that are the same pose
TURN_PX = 25              # image shift caused by one LEFT/RIGHT step (measured: ~25 px)
TURN_FIRST = 0.8          # start turning when the heading is off by at least this many turn-steps (about 20 px)
TURN_KEEP = 0.4           # ...and keep turning the same way until it is below this (stops left/right flip-flop)
STRONG_LINK_INLIERS = 100 # a link this strongly matched is believed even if the matches sit on one big nearby wall
LINK_RETRIES = 4          # if the route keeps sending you through the same link and you keep ending up back at its
                          # entrance, the link does not work in practice: drop it and re-plan
         # turn first when the total turning needed is at least this many steps
CRUMB_CLEAR_MATCHES = 25  # this many matches = sure of the place, forget the trail of key presses
LOST_RETRACE_AFTER = 3    # lost for this many frames in a row: start telling you how to undo your last moves
BEST_RANSAC = 10          # per frame, only this many of the best-looking nodes get the slow geometric check
RECOVER_MIN = 14          # when lost, a node NEAR the last sure position is accepted again with this many matches
FAR_JUMP_MIN = 30         # a node FAR from the last sure position needs this many matches (look-alike walls give ~20-45)...
FAR_JUMP_CONFIRMS = 3     # ...and must be found this many times in a row, close together
LOST_WIDEN = 1.5          # while unsure, the search area around the last sure position grows by this many steps per frame
LOST_WIDEN_MAX = 60       # ...up to this many extra steps
LOST_MAX_NODES = 140      # ...but never checks more than this many nodes per frame (keeps the sim fast)
RECORD_DIR = "runs"       # every run is also saved here (frames + key presses + what the agent thought) so it can be replayed
                          # offline and algorithms can be tested on real runs without the sim. None = off.
NOISE_SCORE = 25.0         # a frame whose pixels look like random static (the very first frame the sim sends) is ignored
START_NODES = 4           # the robot starts where every exploration drive started: the first fix looks at these frames first
LAYOUT_ENABLED = True     # long-range matches (global search, map links, goal) must be spread over most of the view
LAYOUT_BINS = 8           # the view is cut into this many vertical strips...
LAYOUT_BIN_MIN = 3        # ...a strip counts if it holds this many geometrically consistent matches...
LAYOUT_MIN_BINS = 5       # ...and this many strips must count (a single poster fills only 2-3 strips)
LOG_FILE = "tracking_log.csv"   # one line per step; set to None to switch logging off
LOG_IMAGES = 30           # save this many live frames from weak moments into tracking_debug/
LIVE_IMAGE_IS_RGB = False # set True if the sim's camera image is RGB (the saved frames are BGR)
# -----------------------------------------------------------------------------------------------

_SIFT = cv2.SIFT_create()
_BF = cv2.BFMatcher(cv2.NORM_L2)
_EMPTY = (np.zeros((0, 3), np.float32), np.zeros((0, 128), np.float32))


def rootsift(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints (N,3: x, y, size) and RootSIFT descriptors (N,128), same recipe as vlad.py.
    The size of a matched keypoint tells how close the camera is to the thing it sits on."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kp, des = _SIFT.detectAndCompute(gray, None)
    if des is None or len(kp) < 8:
        return _EMPTY
    des = des / np.maximum(des.sum(axis=1, keepdims=True), 1e-9)
    return np.float32([[k.pt[0], k.pt[1], k.size] for k in kp]), np.sqrt(des).astype(np.float32)


_INVERSE = {"FORWARD": "BACKWARD", "BACKWARD": "FORWARD", "LEFT": "RIGHT", "RIGHT": "LEFT"}
_NONE3 = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))


def _ratio_pairs(f1, f2):
    """Cheap first stage: descriptor matches that pass the ratio test (keypoint rows of image 1 and image 2)."""
    (p1, d1), (p2, d2) = f1, f2
    if len(d1) < 8 or len(d2) < 8:
        return _NONE3
    pairs = _BF.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 12:
        return _NONE3
    return p1[[m.queryIdx for m in good]], p2[[m.trainIdx for m in good]]


def _ransac(a, b):
    """Second (slower) stage: keep the matches that agree on one camera geometry."""
    if len(a) < 12:
        return _NONE3
    _, mask = cv2.findFundamentalMat(a[:, :2], b[:, :2], cv2.FM_RANSAC, 3.0, 0.99)
    if mask is None:
        return _NONE3
    keep = mask.ravel().astype(bool)
    return a[keep], b[keep]


def _inlier_points(f1, f2):
    """x/y positions (in image 1 and image 2) of the geometrically consistent SIFT matches."""
    a, b = _ratio_pairs(f1, f2)
    return _ransac(a, b)


def layout_ok(a, b, width: float) -> bool:
    """Do the matches cover most of the view, i.e. several neighbouring wall panels and not one shared poster?
    One poster (the maze reuses them in many places) passes the camera-geometry test on its own, but it
    cannot make its neighbours match too. The view is cut into LAYOUT_BINS vertical strips; a strip counts when
    it holds at least LAYOUT_BIN_MIN matches, and LAYOUT_MIN_BINS strips must count. (Measured on lab mazes with
    known positions: confidently-wrong answers dropped from ~1 in 4 to ~1 in 20, losing few right ones.)"""
    if not LAYOUT_ENABLED:
        return True
    counts = np.histogram(a[:, 0], bins=LAYOUT_BINS, range=(0.0, float(width)))[0]
    return int((counts >= LAYOUT_BIN_MIN).sum()) >= LAYOUT_MIN_BINS


def match(f1, f2) -> tuple[int, float]:
    """(number of geometrically consistent SIFT matches, median horizontal shift f1 - f2 in px)."""
    a, b = _inlier_points(f1, f2)
    if len(a) == 0:
        return 0, 0.0
    return len(a), float(np.median(a[:, 0] - b[:, 0]))


def match_scale(f1, f2, width: float | None = None) -> tuple[int, float, float]:
    """(matches, median horizontal shift, depth cue). Depth cue = median log2(size in image 2 / size in image 1)
    over the matches: positive means image 2 is CLOSER to the scene than image 1. width given = also apply the
    'matches must cover the view' rule."""
    a, b = _inlier_points(f1, f2)
    return _scale_stats(a, b, width)


def _scale_stats(a, b, width: float | None = None) -> tuple[int, float, float]:
    if len(a) == 0:
        return 0, 0.0, 0.0
    if width is not None and not layout_ok(a, b, width):
        return 0, 0.0, 0.0
    if width is not None and len(a) < STRONG_LINK_INLIERS \
            and (np.percentile(a[:, 0], 90) - np.percentile(a[:, 0], 10)) / width < MIN_SPREAD:
        return 0, 0.0, 0.0
    ratio = np.log2(np.maximum(b[:, 2], 1e-3) / np.maximum(a[:, 2], 1e-3))
    return len(a), float(np.median(a[:, 0] - b[:, 0])), float(np.median(ratio))


def match_wide(f1, f2, width: float = 640.0) -> tuple[int, float]:
    """Like match(), but returns 0 when the matches all sit in one small patch of the view (typically a single
    wall texture that also appears somewhere else in the maze) instead of covering a real part of the scene."""
    a, b = _inlier_points(f1, f2)
    if len(a) == 0:
        return 0, 0.0
    if not layout_ok(a, b, width):
        return 0, 0.0
    span = (np.percentile(a[:, 0], 90) - np.percentile(a[:, 0], 10)) / width
    if span < MIN_SPREAD and len(a) < STRONG_LINK_INLIERS:
        return 0, 0.0
    return len(a), float(np.median(a[:, 0] - b[:, 0]))


class TrackingAgent(BaselineAgent):
    def __init__(
        self,
        data_dir: Path,
        *,
        n_clusters: int = 128,
        subsample: int = 1,
        top_k_shortcuts: int = 100,
        cache_dir: Path = Path("cache"),
    ) -> None:
        # these must exist before BaselineAgent.__init__ runs (it calls _build_graph)
        self._cache_dir = Path(cache_dir)
        self._store: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._store_loaded = False
        self._store_dirty = False
        self._mode = "track"
        self._rot = 0
        self._move = None
        self._logged_step = None
        self._window = max(3, WINDOW // subsample)
        self.lost = True
        self.weak = 0
        self.match_inliers = 0
        self.align_dx = 0.0
        self.turn_need = 0.0
        self._n = 0
        self._live_width = 640.0
        self._steer = None
        self.depth_steps = 0.0
        self._px_samples: list[float] = []
        self._fwd_hops: list[int] = []
        self._pt = None
        self._force_deep = False
        self._h_down = False
        self._notice, self._notice_n = "", 0
        self._shown, self._cand, self._cand_n = "", "", 0
        self._last_good = None
        self._pending = None
        self._rec = None
        self._pending_n = 0
        self._crumbs: list[str] = []      # your key presses since the last place we were sure about
        self._lost_ticks = 0
        self._turn_side = 0
        self._link_arrivals: dict[tuple[int, int], int] = {}
        self._prev_node = None
        self._blocked = 0
        self._seen_step = None
        self._prev_gray = None
        self._last_name = ""
        self._turning = False       # was the last real key a pure left/right turn?
        self._turning_prev = False
        self._saved = 0
        self._log = None
        super().__init__(
            data_dir,
            n_clusters=n_clusters,
            subsample=subsample,
            top_k_shortcuts=top_k_shortcuts,
            cache_dir=cache_dir,
        )

    # ------------------------------------------------------------------ helpers
    def _node_feats(self, i: int):
        """RootSIFT keypoints/descriptors of a database frame (cached on disk after the first run)."""
        if not self._store_loaded:
            self._store_loaded = True
            path = self._cache_dir / f"rootsift3_feats_{len(self.frames)}.pkl"
            if path.exists():
                try:
                    self._store = pickle.loads(path.read_bytes())
                except Exception:
                    self._store = {}
        key = str(self.frames[i].path)
        item = self._store.get(key)
        if item is None:
            image = cv2.imread(key)
            pts, des = rootsift(image) if image is not None else _EMPTY
            item = (pts, des.astype(np.float16))
            self._store[key] = item
            self._store_dirty = True
        return item[0], item[1].astype(np.float32)

    def _save_store(self) -> None:
        if self._store_dirty:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            (self._cache_dir / f"rootsift3_feats_{len(self.frames)}.pkl").write_bytes(pickle.dumps(self._store))
            self._store_dirty = False

    def _image(self, node: int) -> np.ndarray:
        cache = self.__dict__.setdefault("_img_cache", {})
        if node not in cache:
            if len(cache) > 400:
                cache.clear()
            cache[node] = super()._image(node)
        return cache[node]

    def _window_nodes(self, centre: int) -> dict[int, int]:
        """Every node within a few graph steps of the last position (-> its distance in steps),
        including the other side of any verified link, so crossing onto another drive's path is
        tracked instead of being mistaken for a jump."""
        extra = int(min(LOST_WIDEN_MAX, LOST_WIDEN * self._lost_ticks))
        found = dict(nx.single_source_shortest_path_length(self.graph, centre, cutoff=self._window + extra))
        if len(found) > LOST_MAX_NODES:                    # keep the closest ones only
            found = dict(sorted(found.items(), key=lambda kv: kv[1])[:LOST_MAX_NODES])
        return found

    def _hop_penalty(self) -> float:
        return TURN_HOP_PENALTY if self._turning else HOP_PENALTY

    def _best_of(self, qf, nodes: list[int], hops: dict[int, int] | None = None, wide: bool = False, lenient=()):
        """Best node by SIFT matches (nodes far from the last position need slightly more), plus the best
        competitor that is not just a neighbour of the winner. Returns (node, matches, shift, rival).
        Every node gets the cheap descriptor test; only the BEST_RANSAC most promising ones (and the current
        node) get the slow geometric check -- that is what keeps the sim running at full speed."""
        hops = hops or {}
        pre = []
        for j in dict.fromkeys(nodes):
            a, b = _ratio_pairs(qf, self._node_feats(j))
            pre.append((len(a), j, a, b))
        chosen = {t[1] for t in sorted(pre, key=lambda t: -t[0])[:BEST_RANSAC]}
        if self.current is not None:
            chosen.add(self.current)
        width = self._live_width if wide else None
        lenient = set(lenient)
        scored = []
        for cnt, j, a, b in pre:
            if j in chosen and cnt >= 12:
                inl, dx, ls = _scale_stats(*_ransac(a, b), None if j in lenient else width)
            else:
                inl, dx, ls = 0, 0.0, 0.0
            # prefer the node you are at the SAME DISTANCE from: matches count less if the keypoints are bigger/smaller
            fit = inl * float(np.exp(-0.5 * (ls / SCALE_SIGMA) ** 2)) if inl >= ACCEPT_TRACK else inl
            scored.append((fit - self._hop_penalty() * hops.get(j, 0), inl, dx, j))
        _, inl, dx, best = max(scored, key=lambda t: (t[0], -abs(t[3] - (self.current if self.current is not None else t[3]))))
        rival = max((t[1] for t in scored if abs(t[3] - best) > 3), default=0)
        return best, inl, dx, rival

    # ------------------------------------------------------------------ graph
    def _build_graph(self, bounds, top_k: int) -> nx.DiGraph:
        self._starts = [int(s0) for s0, _ in bounds]
        n = len(self.database)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))

        # 1. recorded steps. Walking BACKWARD along the recording is blind, so it costs more.
        for start, end in bounds:
            for i in range(start, end - 1):
                act = self.frames[i].action
                fw, bw = TEMPORAL_WEIGHT, TEMPORAL_WEIGHT
                if act == "FORWARD":
                    bw = BACKWARD_COST
                elif act == "BACKWARD":
                    fw = BACKWARD_COST
                graph.add_edge(i, i + 1, weight=fw)
                graph.add_edge(i + 1, i, weight=bw)

        # 2. "same pose again" links: the explorer wiggled (turn right 6, turn left 6) and ended with the
        #    same view. Joining those frames stops the route from replaying the wiggle.
        rot_p = np.zeros(n + 1, int)
        fwd_p = np.zeros(n + 1, int)
        for k, f in enumerate(self.frames):
            rot_p[k + 1] = rot_p[k] + (f.action == "RIGHT") - (f.action == "LEFT")
            fwd_p[k + 1] = fwd_p[k] + (f.action == "FORWARD") - (f.action == "BACKWARD")
        same = 0
        for start, end in bounds:
            for i in range(start, end):
                for j in range(i + 2, min(end, i + SAME_POSE_SPAN + 1)):
                    if rot_p[j] != rot_p[i] or fwd_p[j] != fwd_p[i]:
                        continue
                    inl, dx = match(self._node_feats(i), self._node_feats(j))
                    if inl >= SAME_POSE_MIN_INLIERS and abs(dx) <= SAME_POSE_MAX_SHIFT:
                        graph.add_edge(i, j, weight=SAME_POSE_WEIGHT, visual=True, same=True, dx=dx)
                        graph.add_edge(j, i, weight=SAME_POSE_WEIGHT, visual=True, same=True, dx=-dx)
                        same += 1
        print(f"same-pose links: {same}")

        # 3. verified visual shortcuts between different places in the maze / different drives
        probe = cv2.imread(str(self.frames[0].path))
        self._width = float(probe.shape[1]) if probe is not None else 640.0
        similarity = self.database @ self.database.T
        for i in range(n):
            similarity[i, max(0, i - MIN_SHORTCUT_GAP) : i + MIN_SHORTCUT_GAP + 1] = -2
        similarity[np.tril_indices(n)] = -2
        flat = similarity.ravel()
        pool = int(min(flat.size, top_k * SHORTCUT_POOL))
        cand = np.argpartition(flat, -pool)[-pool:]
        cand = cand[np.argsort(-flat[cand])]
        chosen: list[tuple[int, int]] = []
        for index in cand:
            if flat[index] <= -1:
                break
            i, j = divmod(int(index), n)
            if any(abs(i - a) <= SHORTCUT_SUPPRESS and abs(j - b) <= SHORTCUT_SUPPRESS for a, b in chosen):
                continue
            inl, dx = match_wide(self._node_feats(i), self._node_feats(j), self._width)
            if inl < SHORTCUT_MIN_INLIERS:
                continue
            w = VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * _distance(float(flat[index]))
            graph.add_edge(i, j, weight=w, visual=True, dx=dx)
            graph.add_edge(j, i, weight=w, visual=True, dx=-dx)
            chosen.append((i, j))
            # Where two drives overlap, the neighbouring frames match too. Link those as well, otherwise one step
            # off the single linked frame the route says "go back to that frame first" (forward/back jitter).
            for k in range(-LINK_SPREAD_STEPS, LINK_SPREAD_STEPS + 1):
                for a, b in ((i + k, j + k), (i + k, j - k)):
                    if k == 0 or not (0 <= a < n and 0 <= b < n) or a == b or graph.has_edge(a, b):
                        continue
                    if self.frames[a].trajectory == self.frames[b].trajectory and abs(a - b) < MIN_SHORTCUT_GAP:
                        continue
                    inl2, dx2 = match_wide(self._node_feats(a), self._node_feats(b), self._width)
                    if inl2 >= LINK_EXTRA_MIN_INLIERS:
                        w2 = VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * _distance(float(self.database[a] @ self.database[b]))
                        graph.add_edge(a, b, weight=w2, visual=True, dx=dx2)
                        graph.add_edge(b, a, weight=w2, visual=True, dx=-dx2)
            if len(chosen) >= top_k:
                break
        self._save_store()
        print(f"graph: {n} nodes, {graph.number_of_edges()} directed edges "
              f"({same} same-pose links, {len(chosen)} verified shortcuts)")
        return graph

    # ------------------------------------------------------------------ Agent
    def setup(self, info: SessionInfo) -> None:
        super().setup(info)
        self._target_imgs = list(getattr(info, "targets", []) or [])
        self.lost = True
        self.weak = 0
        self.match_inliers = 0
        self.align_dx = 0.0

    def _localise(self, image: np.ndarray) -> None:
        if LIVE_IMAGE_IS_RGB:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self._live_width = float(image.shape[1])
        qf = rootsift(image)
        qv_box: list = []

        def qvec():
            if not qv_box:
                vl = getattr(self.extractor, "_vlad", None)
                qv_box.append(vl(qf[1] if len(qf[1]) else None) if vl is not None else self.extractor.extract(image))
            return qv_box[0]

        self._lost_ticks = self._lost_ticks + 1 if (self.lost or self.weak > 0) else 0
        best, inl, dx, mode = None, 0, 0.0, "track"
        reach = self._window_nodes(self.current) if self.current is not None else {}
        deep_done = False
        if self._force_deep:                        # you asked for the slow, thorough check
            self._force_deep = False
            order = [int(j) for j in np.argsort(-(self.database @ qvec()))[:DEEP_TOPK]]
            cand, cinl, cdx, _ = self._best_of(qf, order + list(reach), reach, wide=True)
            if cinl >= ACCEPT_TRACK:
                best, inl, dx, mode, deep_done = cand, cinl, cdx, "deep", True
                self.lost, self.weak, self._pending = False, 0, None
                self._notice = f"deep check: node {cand} ({cinl} matches)"
            else:
                self._notice = f"deep check: nothing convincing ({cinl} matches) -- look around / keep going"
            self._notice_n = NOTICE_TICKS
        if not deep_done and self.current is not None and self._last_good is not None:
            best, inl, dx, rival = self._best_of(qf, list(reach), reach)
            if self.lost:                           # unsure: look again around where we last were sure, over a growing area
                trusted = inl >= RECOVER_MIN and inl >= SECOND_RATIO * rival
                if trusted:
                    self.lost, mode = False, "recover"
            else:
                trusted = inl >= STRONG_MATCH or (inl >= ACCEPT_TRACK and inl >= SECOND_RATIO * rival)
            if trusted:
                self.weak = 0
                self._pending = None
                if best != self.current:            # do not swap places for a marginal gain: stay unless clearly better
                    cur_inl, cur_dx = match(qf, self._node_feats(self.current))
                    if cur_inl >= ACCEPT_TRACK and inl < cur_inl + max(SWITCH_MARGIN_MIN, SWITCH_MARGIN_FRAC * cur_inl):
                        best, inl, dx = self.current, cur_inl, cur_dx
            elif self.lost:
                best = None
            else:
                self.weak += 1
                if self.weak < LOST_PATIENCE:       # one or two doubtful frames: keep the old position
                    best, dx, mode = self.current, 0.0, "weak"
                else:
                    best = None
        if best is None and not deep_done:          # first fix, or lost: search the whole map
            top = [int(j) for j in np.argsort(-(self.database @ qvec()))[:GLOBAL_TOPK]]
            if self._last_good is None:             # first fix: we are at the spawn, where every drive began
                top = [s0 + k for s0 in getattr(self, "_starts", []) for k in range(START_NODES)] + top
            first = self._last_good is None
            cand, cinl, cdx, _ = self._best_of(qf, top + list(reach), reach, wide=True,
                                               lenient=[s0 + k for s0 in getattr(self, '_starts', []) for k in range(START_NODES)] if first else ())
            mode = "global"
            far = self._last_good is not None and cand not in reach
            believed = cinl >= (FAR_JUMP_MIN if far else ACCEPT_GLOBAL)
            if believed and far:
                # a far-away jump: only believe it if the SAME place is found several times in a row
                if self._pending is not None and abs(cand - self._pending) <= JUMP_CONFIRM_RADIUS:
                    self._pending_n += 1
                    if self._pending_n >= FAR_JUMP_CONFIRMS:
                        self._pending, self._pending_n = None, 0
                    else:
                        believed = False
                else:
                    self._pending, self._pending_n, believed = cand, 1, False
            elif believed:
                self._pending = None
            if believed:
                best, inl, dx = cand, cinl, cdx
                self.lost, self.weak = False, 0
            else:                                   # not sure: keep showing the last place we were sure about
                best = self._last_good if self._last_good is not None else top[0]
                inl, dx = 0, 0.0
                self.lost, mode = True, "lost"
        if not self.lost:
            self._last_good = int(best)
            if inl >= CRUMB_CLEAR_MATCHES:         # sure about the place again: the trail since the last sure place is no longer needed
                self._crumbs.clear()
        self._calibrate(self._last_name, int(best), int(inl), float(dx), mode, reach)
        # how far in front of / behind the chosen node are we? (+ = the node is further along: we are behind it)
        self.depth_steps = 0.0
        if not self.lost and inl >= ACCEPT_TRACK:
            _, _, ls = match_scale(qf, self._node_feats(int(best)))
            self.depth_steps = float(np.clip(ls / SCALE_PER_STEP, -DEPTH_CLIP, DEPTH_CLIP))

        self.current = int(best)
        self.match_inliers = int(inl)
        self.align_dx = float(dx)
        if qv_box:
            self.match_distance = _distance(float(self.database[self.current] @ qv_box[0]))
        try:
            self.path = nx.shortest_path(self.graph, self.current, self.goal, weight="weight")
        except nx.NetworkXNoPath:
            self.path = [self.current]
        self._drop_failing_link()
        self._steer = self._compute_steer(qf)
        if self._notice_n > 0:
            self._notice_n -= 1
        self._update_hint()
        self._panel = self._render_panel()
        self._mode = mode

    @staticmethod
    def _is_static(image: np.ndarray) -> bool:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        g = g.astype(np.float32)
        return float(np.abs(g - cv2.blur(g, (3, 3))).mean()) > NOISE_SCORE

    def act(self, obs):
        if self._is_static(obs.image):         # not a real camera frame (random static): wait for the next one
            return WAIT
        new_step = obs.step != self._seen_step
        self._seen_step = obs.step
        self._turning = self._turning_prev
        try:                                         # deep-check key (edge triggered)
            keys = self.viewer.keys() if getattr(self, "viewer", None) is not None else ()
            down = DEEP_KEY is not None and DEEP_KEY in keys
            if down and not self._h_down:
                self._force_deep = True
                self._localised_step = None          # make sure this tick is processed again with the deep search
            self._h_down = down
        except Exception:
            pass
        gray = cv2.cvtColor(obs.image, cv2.COLOR_BGR2GRAY) if obs.image.ndim == 3 else obs.image
        gray = cv2.resize(gray, (80, 60)).astype(np.float32)
        if not new_step:
            pass
        elif self._last_name in ("FORWARD", "BACKWARD") and self._prev_gray is not None:
            # we pressed a move key last tick: if the picture did not change at all, we pushed against a wall
            self._blocked = self._blocked + 1 if float(np.abs(gray - self._prev_gray).mean()) < BLOCKED_DIFF else 0
        elif self._last_name in ("LEFT", "RIGHT"):
            self._blocked = 0
        if new_step:
            self._prev_gray = gray
        action = self._decide(super().act(obs))
        name = getattr(action[0], "name", "") if isinstance(action, tuple) else ""
        if new_step:
            self._last_name = name
            if name in _INVERSE and not getattr(self, "_skip_crumb", False):   # remember the moves, so that when lost we can say how to undo them
                if self._crumbs and self._crumbs[-1] == _INVERSE[name]:
                    self._crumbs.pop()
                else:
                    self._crumbs = (self._crumbs + [name])[-80:]
        if name and name != "IDLE":            # pauses keep the previous state
            self._turning_prev = name in ("LEFT", "RIGHT")
        if obs.step != self._logged_step:
            self._logged_step = obs.step
            self._record(obs.image, obs.step, action)
        return action

    def _decide(self, action):
        """Hook: what to send to the robot. Here: whatever the keyboard says (you drive). auto_agent.py overrides it."""
        return action

    def _plan(self) -> tuple[float, str | None, int]:
        """Net RIGHT turns needed before the route's next real move, what that move is, and how many times
        in a row it is repeated. A visual link means "you are at the same place as that other frame": nothing
        to walk, but the two views may be rotated against each other, so its shift is added to the turning."""
        rot, n = 0.0, 0
        move, run = None, 0
        for a, b in zip(self.path[:-1], self.path[1:]):
            edge = self.graph[a][b]
            if edge.get("visual"):
                if move:
                    break
                rot += edge.get("dx", 0.0) / TURN_PX
            else:
                act = self._edge_action(a, b)
                if move:
                    if act != move:
                        break
                    run += 1
                elif act in ("FORWARD", "BACKWARD"):
                    move, run = act, 1
                elif act == "RIGHT":
                    rot += 1
                elif act == "LEFT":
                    rot -= 1
            n += 1
            if n >= LOOKAHEAD_MAX:
                break
        return rot, move, run

    def _drop_failing_link(self) -> None:
        """The route says 'cross link A->B from here'. If the user keeps arriving at A again and again and never
        gets across, the link is not usable in practice (the two recorded poses differ more than the match said).
        After LINK_RETRIES arrivals, remove it and re-plan."""
        if self.current != self._prev_node and len(self.path) >= 2 and self.graph.has_edge(self.path[0], self.path[1]) \
                and self.graph[self.path[0]][self.path[1]].get("visual"):
            key = (self.path[0], self.path[1])
            self._link_arrivals[key] = self._link_arrivals.get(key, 0) + 1
            if self._link_arrivals[key] >= LINK_RETRIES:
                a, b = key
                self.graph.remove_edge(a, b)
                if self.graph.has_edge(b, a):
                    self.graph.remove_edge(b, a)
                self._link_arrivals.pop(key, None)
                try:
                    self.path = nx.shortest_path(self.graph, self.current, self.goal, weight="weight")
                except nx.NetworkXNoPath:
                    self.path = [self.current]
        self._prev_node = self.current

    def _compute_steer(self, qf):
        """Visual steering. Compare the LIVE view with route frames further ahead (farthest one that shares enough
        features with what you see). Two things come out of that match, both measured in YOUR view:
          dx = how far sideways that frame sits (turn until it is ~0),
          ls = log2(size of its features / size of yours): > 0 it is closer to the scene than you (go FORWARD),
               < 0 it is farther (go BACKWARD).
        None if no route frame is in view (then the recorded-action logic below takes over)."""
        if not STEER_ENABLED or self.lost or len(self.path) < 2:
            return None
        route = self.path[1 : 1 + STEER_LOOKAHEAD]
        tried = set()
        for pos in STEER_TARGETS:
            idx = min(pos, len(route)) - 1
            if idx < 0 or idx in tried:
                continue
            tried.add(idx)
            inl, dx, ls = match_scale(qf, self._node_feats(route[idx]), self._live_width)
            if inl >= STEER_MIN_INLIERS:
                return dx, ls, route[idx], inl
        return None

    @property
    def px_per_press(self) -> float:
        """Measured shift (px) of the picture per LEFT/RIGHT key press (median of recent presses)."""
        n = len(self._px_samples)
        med = float(np.median(self._px_samples)) if n else DEFAULT_PX_PER_PRESS
        w = n / (n + CAL_PRIOR)
        return float(np.clip(w * med + (1 - w) * DEFAULT_PX_PER_PRESS, 5.0, 30.0))

    @property
    def nodes_per_press(self) -> float:
        """Measured recorded-steps covered per FORWARD key press."""
        n = len(self._fwd_hops)
        mean = float(np.mean(self._fwd_hops)) if n else DEFAULT_NODES_PER_PRESS
        w = n / (n + CAL_PRIOR)
        return float(np.clip(w * mean + (1 - w) * DEFAULT_NODES_PER_PRESS, 0.12, 1.0))

    def _calibrate(self, last: str, best: int, inl: int, dx: float, mode: str, reach: dict) -> None:
        """Learn the size of one key press from how the tracker moved after the previous key."""
        prev = self._pt
        if prev and mode == "track" and prev["mode"] == "track" and inl >= 30 and prev["inl"] >= 30:
            if last in ("LEFT", "RIGHT") and best == prev["node"]:
                shift = (dx - prev["dx"]) * (1 if last == "LEFT" else -1)    # LEFT makes the shift grow, RIGHT shrinks it
                if 3.0 <= shift <= 40.0:
                    self._px_samples = (self._px_samples + [shift])[-60:]
            elif last == "FORWARD" and self._blocked == 0 and best in reach and reach[best] <= 4:
                self._fwd_hops = (self._fwd_hops + [reach[best]])[-120:]
        self._pt = {"node": best, "dx": dx, "inl": inl, "mode": mode}

    def _turn_words(self, need: float) -> str | None:
        """RIGHT/LEFT text if the heading error is big enough (with hysteresis), else None."""
        side = 1 if need > 0 else -1
        limit = TURN_KEEP if self._turn_side == side else TURN_FIRST
        if abs(need) < limit:
            self._turn_side = 0
            return None
        self._turn_side = side
        deg = abs(need) * TURN_PX * DEG_PER_PX
        presses = max(1, round(abs(need) * TURN_PX / self.px_per_press))     # in YOUR key presses
        return f"{'RIGHT' if side > 0 else 'LEFT'} (about {presses} press{'' if presses == 1 else 'es'}, ~{deg:.0f} deg)"

    def _hint(self) -> str:
        return self._shown if self._shown else self._raw_hint()

    @staticmethod
    def _kind(text: str) -> str:
        return text.split()[0] if text.startswith(("LEFT", "RIGHT", "FORWARD", "BACKWARD")) else text

    _OPPOSITE = {("LEFT", "RIGHT"), ("RIGHT", "LEFT"), ("FORWARD", "BACKWARD"), ("BACKWARD", "FORWARD")}

    def _update_hint(self) -> None:
        """Show the newest advice right away. Only a flip to the OPPOSITE direction (left<->right, forward<->back)
        has to be seen HINT_HOLD times in a row first, so a single noisy frame cannot make it flip-flop."""
        raw = self._raw_hint()
        if not self._shown or raw.startswith(("blocked", "at the goal", "not sure", "no known route", "lost")):
            self._shown, self._cand, self._cand_n = raw, raw, 0
            return
        kr, ks = self._kind(raw), self._kind(self._shown)
        if kr == ks or (kr, ks) not in self._OPPOSITE:
            self._shown, self._cand, self._cand_n = raw, raw, 0
            return
        if kr == self._kind(self._cand):
            self._cand_n += 1
        else:
            self._cand, self._cand_n = raw, 1
        if self._cand_n >= HINT_HOLD:
            self._shown, self._cand_n = raw, 0

    def _next_turn(self):
        """Look DECISION_MOVES real moves down the route. Returns (net_turn, moves_until_turn, first_move):
        net_turn = RIGHT-positive turn-steps (turn actions and link rotations) accumulated over that stretch,
        moves_until_turn = how many real moves before that turning reaches QUIET_TURN (0 = right now)."""
        cum, moves, first, until = 0.0, 0, None, None
        for k, (a, b) in enumerate(zip(self.path[:-1], self.path[1:])):
            if k >= LOOKAHEAD_MAX or moves >= DECISION_MOVES:
                break
            edge = self.graph[a][b]
            if edge.get("visual"):
                cum += edge.get("dx", 0.0) / TURN_PX
            else:
                act = self._edge_action(a, b)
                if act == "RIGHT":
                    cum += 1
                elif act == "LEFT":
                    cum -= 1
                elif act in ("FORWARD", "BACKWARD"):
                    first = first or act
                    moves += 1
            if until is None and abs(cum) >= QUIET_TURN:
                until = moves
        return cum, (until if until is not None else None), first

    def _raw_hint(self) -> str:
        self._rot, self._move = 0, None
        if self._blocked >= BLOCKED_COUNT:
            return "blocked by a wall -- turn and look around"
        if len(self.path) < 2 and not self.lost:
            return "at the goal" if self.current == self.goal else "no known route from here -- look around"
        if self.lost:
            if self._lost_ticks >= LOST_RETRACE_AFTER and self._crumbs:
                top = self._crumbs[-1]
                run = 0
                for c in reversed(self._crumbs):
                    if c != top:
                        break
                    run += 1
                more = len(self._crumbs) - run
                return (f"lost -- retrace your steps: {_INVERSE[top]} x{run}"
                        + (f", then {more} more move{'' if more == 1 else 's'}" if more else ""))
            return "not sure where you are -- turn and look around"
        if self._steer is not None:
            dx, ls, target, _ = self._steer
            self.turn_need = dx / TURN_PX
            side = 1 if dx > 0 else -1
            if abs(dx) >= (STEER_KEEP_PX if self._turn_side == side else STEER_TURN_PX):
                self._turn_side, self._move = side, None
                presses = max(1, round(abs(dx) / self.px_per_press))
                return (f"{'RIGHT' if side > 0 else 'LEFT'} (about {presses} press{'' if presses == 1 else 'es'}, "
                        f"~{abs(dx) * DEG_PER_PX:.0f} deg)")
            self._turn_side = 0
            move = "BACKWARD" if ls < -STEER_DEPTH_MIN else "FORWARD"
            self._move = move
            if len(self.path) <= 4 and abs(ls) < STEER_DEPTH_MIN:
                return "press space if the view matches the goal"
            if move == "FORWARD" and HINT_STYLE == "decision" and len(self.path) > 4:
                cum, until, first = self._next_turn()          # the 'turn coming up' countdown you liked
                if first == "FORWARD" and until is not None:
                    until_adj = max(0, int(round(until + self.depth_steps)))
                    if until_adj > 0:
                        presses = max(1, round(until_adj / self.nodes_per_press))
                        return f"FORWARD (then {'RIGHT' if cum > 0 else 'LEFT'} in about {presses} press{'' if presses == 1 else 'es'})"
            return move
        if HINT_STYLE == "decision" and len(self.path) > 4:
            cum, until, first = self._next_turn()
            align = self.align_dx / TURN_PX
            if first and until is None and abs(align) < QUIET_ALIGN:
                self._move, self.turn_need = first, 0.0       # straight stretch: no fiddly corrections
                return first
            until_adj = None
            if until is not None:     # the node is not exactly where you stand: shift the countdown by the depth cue
                until_adj = max(0, int(round(until + (self.depth_steps if first == "FORWARD" else -self.depth_steps))))
            if first and until_adj is not None and until_adj > 0 and abs(align) < QUIET_ALIGN:
                self._move, self.turn_need = first, 0.0       # turn is coming, not here yet: say so, keep walking
                side = "RIGHT" if cum > 0 else "LEFT"
                presses = max(1, round(until_adj / self.nodes_per_press))          # in YOUR forward presses
                return f"{first} (then {side} in about {presses} press{'' if presses == 1 else 'es'})"
        # near a turn or junction (or no straight run to follow): give the full turning advice
        rot, move, run = self._plan()
        self._rot, self._move = rot, move
        align = self.align_dx / TURN_PX
        if move and run < SHORT_RUN and abs(align) < SHORT_RUN_MAX_ALIGN and abs(rot) < TURN_FIRST:
            align = 0.0     # a short hop with no turn pending: just take it (when a turn IS pending, align is how far you have already turned)
        # live content shifted right of the node's view => turned too far left => needs RIGHT turns
        self.turn_need = align + rot
        words = self._turn_words(self.turn_need)
        if words:
            return words
        if move in ("FORWARD", "BACKWARD"):
            return move
        return "press space if the view matches the goal"

    def hud(self) -> list[str]:
        lines = super().hud()
        if self.current is not None:
            state = "LOST" if self.lost else ("weak" if self.weak else "tracking")
            lines.append(f"SIFT matches {self.match_inliers} | shift {self.align_dx:+.0f}px | "
                         f"depth {self.depth_steps:+.1f} | turn need {self.turn_need:+.1f} | {state}")
        if self._notice_n > 0:
            lines.append(self._notice)
        elif DEEP_KEY:
            lines.append(f"press {DEEP_KEY.upper()} for a thorough position check | 1 turn press = {self.px_per_press:.0f}px, "
                         f"1 forward press = {self.nodes_per_press:.2f} steps")
        return lines

    # ------------------------------------------------------------------ logging for diagnosis
    def _save_run(self, image: np.ndarray, tick, action) -> None:
        """Append one step (the live frame as a JPEG, the key pressed, our belief) to runs/run_<time>.pkl."""
        try:
            if self._rec is None:
                Path(RECORD_DIR).mkdir(exist_ok=True)
                self._rec = open(Path(RECORD_DIR) / time.strftime("run_%Y%m%d_%H%M%S.pkl"), "wb")
                targets = [cv2.imencode(".jpg", t)[1].tobytes() for t in getattr(self, "_target_imgs", [])]
                pickle.dump({"header": True, "goal": self.goal, "frames": len(self.frames), "targets": targets}, self._rec)
            act = action[0] if isinstance(action, tuple) else action
            cnt = action[1] if isinstance(action, tuple) and len(action) > 1 else 1
            ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            pickle.dump({"n": self._n, "tick": tick, "key": getattr(act, "name", str(act)), "count": cnt,
                         "jpg": jpg.tobytes(), "node": self.current, "mode": self._mode, "matches": self.match_inliers,
                         "hint": self._hint(), "lost": bool(self.lost)}, self._rec)
            self._rec.flush()
        except Exception as exc:          # recording must never break a run
            if not getattr(self, "_rec_warned", False):
                print("run recording failed:", exc)
                self._rec_warned = True

    def _record(self, image: np.ndarray, tick, action) -> None:
        self._n += 1
        if RECORD_DIR is not None:
            self._save_run(image, tick, action)
        if LOG_FILE is None:
            return
        if self._log is None:
            self._file = open(LOG_FILE, "w", newline="")
            self._log = csv.writer(self._file)
            self._log.writerow(["n", "tick", "node", "drive", "mode", "matches", "dx", "lost", "hops",
                                "rot", "move", "turn_need", "hint", "action", "path_head", "target", "target_matches", "depth", "px_per_press", "nodes_per_press"])
        hint = self._hint()
        self._log.writerow([self._n, tick, self.current, self.frames[self.current].trajectory, self._mode,
                            self.match_inliers, round(self.align_dx, 1), int(self.lost),
                            len(self.path) - 1, round(self._rot, 2), self._move, round(self.turn_need, 2), hint,
                            action, " ".join(map(str, self.path[:6])),
                            self._steer[2] if self._steer else "", self._steer[3] if self._steer else "",
                            round(self.depth_steps, 2), round(self.px_per_press, 1), round(self.nodes_per_press, 2)])
        self._file.flush()
        if self.match_inliers < ACCEPT_TRACK and self._saved < LOG_IMAGES:
            Path("tracking_debug").mkdir(exist_ok=True)
            cv2.imwrite(f"tracking_debug/n{self._n:05d}_live.png", image)
            node = cv2.imread(str(self.frames[self.current].path))
            if node is not None:
                cv2.imwrite(f"tracking_debug/n{self._n:05d}_node{self.current}.png", node)
            self._saved += 1


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", help="exploration data directory (default: download)")
    parser.add_argument("--subsample", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size")
    parser.add_argument("--top-k", type=int, default=100, help="visual shortcut edges")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = TrackingAgent(
        cli.exploration_data(args, args.data),
        n_clusters=args.n_clusters,
        subsample=args.subsample,
        top_k_shortcuts=args.top_k,
        cache_dir=Path("cache") / challenge.replace(":", "-"),
    )
    run(agent, challenge, **cli.run_options(args))
