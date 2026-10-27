"""
Goal candidate picker: the goal vote can pick a look-alike (maze 21: it chose node 2069, the real goal
was its runner-up 2302 / the VLAD guess 2305), but a person sees the difference in seconds -- the real
place matches a whole run of neighbouring wall panels, a look-alike only one repeated poster.

At setup the agent lists the vote's best clusters plus the VLAD guess. Each candidate is refined to the
frame within +-3 steps that best matches one goal view (phase B's matcher), shown in the panel strip
beside that goal view, and drawn into logs/goal_candidates.png (opened automatically on Windows).
Number keys 1..5 choose (any time during the run, not only at the start); the choice is remembered in
cache/<challenge>/goal_candidates_<key>.json, so attempts 2-3 start on the goal you picked.

Pure helpers here; FinalRouteAgent wires them in (keys, HUD, panel).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_CANDIDATES = 5
MIN_VOTE_SCORE = 10.0     # vote clusters below this are noise (maze 21: 2.4 was a single poster)
SAME_PLACE = 20           # a VLAD guess this close to a vote cluster (same drive) is the same candidate
REFINE = 3                # look for the best-matching frame within +-this many steps (same drive)
VIEW_NAMES = ["FRONT", "LEFT", "BACK", "RIGHT"]

# spellings a viewer may use for the number keys; unknown key names get printed once
_DIGIT_PREFIXES = ("", "kp", "kp_", "keypad", "keypad_", "num", "num_", "numpad", "numpad_")


def digit_of_any(keys) -> int | None:
    """digit_of over everything the viewer's Keys object offers: held now, or tapped since the last poll
    (a quick tap can fall between two polls of the held set)."""
    d = digit_of(keys)
    if d is not None:
        return d
    for name in ("tapped", "held"):
        part = getattr(keys, name, None)
        try:
            part = part() if callable(part) else part
        except Exception:
            part = None
        if part is not None:
            d = digit_of(part)
            if d is not None:
                return d
    return None


def digit_of(keys) -> int | None:
    """1..9 if a number key (top row or keypad) is held, else None. ``keys`` only needs ``in``:
    the viewer's Keys object supports membership tests but is not iterable."""
    for d in range(1, 10):
        for name in [f"{p}{d}" for p in _DIGIT_PREFIXES] + [f"[{d}]"]:
            try:
                if name in keys:
                    return d
            except Exception:
                return None
    return None


def is_noise(image) -> bool:
    """Random static instead of a photo. The SDK's pre-flight check calls setup() with noise goal views."""
    try:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        g = g.astype(np.float32)
        return float(np.abs(g - cv2.blur(g, (3, 3))).mean()) > 25.0   # tracking_agent.NOISE_SCORE
    except Exception:
        return False


@dataclass
class Candidate:
    node: int            # the frame to route to (refined)
    source: str          # "vote #1 (score 63)" / "VLAD guess"
    view: int | None     # goal view it matched best
    inliers: int
    parallax: float

    def label(self, i: int) -> str:
        v = VIEW_NAMES[self.view] if self.view is not None and self.view < 4 else "?"
        return f"{i + 1}: node {self.node} ~ goal {v} view, {self.inliers} inl ({self.source})"


def cache_key(targets, n_frames: int, links_version: int) -> str:
    """Same recipe as route_agent's goal-vote cache, so one goal photo set = one key."""
    blob = b"".join(cv2.imencode(".png", t)[1].tobytes() for t in targets)
    return hashlib.md5(blob + f"{n_frames}-{links_version}".encode()).hexdigest()[:16]


def load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_cache(path: Path, top, choice: int | None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"top": [[int(n), float(s)] for n, s in top], "choice": choice}))
    except Exception as exc:
        print("could not save goal candidates:", exc)


def build(top, vlad: int | None, frames, image_of, matcher) -> list[Candidate]:
    """Vote clusters (score >= MIN_VOTE_SCORE, best first) + the VLAD guess, each refined."""
    raw: list[tuple[int, str]] = []
    for rank, (node, score) in enumerate(top or []):
        if score >= MIN_VOTE_SCORE:
            raw.append((int(node), f"vote #{rank + 1}, score {score:.0f}"))
    same = lambda a, b: abs(a - b) <= SAME_PLACE and frames[a].trajectory == frames[b].trajectory  # noqa: E731
    if vlad is not None and vlad >= 0 and not any(same(vlad, n) for n, _ in raw):
        raw.append((int(vlad), "VLAD guess"))
    if not raw and vlad is not None and vlad >= 0:
        raw.append((int(vlad), "VLAD guess"))
    out: list[Candidate] = []
    for node, source in raw[:MAX_CANDIDATES]:
        best = Candidate(node, source, None, 0, 0.0)
        drive = frames[node].trajectory
        for j in range(max(0, node - REFINE), min(len(frames), node + REFINE + 1)):
            if frames[j].trajectory != drive:
                continue
            est = matcher.estimate(image_of(j))
            if est is not None and est.inliers > best.inliers:
                best = Candidate(j, source, int(est.view), int(est.inliers), float(est.parallax))
        out.append(best)
    return out


def montage(cands: list[Candidate], targets, image_of, chosen: int) -> np.ndarray:
    """One picture to decide from: the four goal views on top, then per candidate its frame beside
    the goal view it matched."""
    h, w = 240, 320
    fit = lambda im: cv2.resize(im, (w, h)) if im is not None else np.zeros((h, w, 3), np.uint8)  # noqa: E731
    font = cv2.FONT_HERSHEY_SIMPLEX
    top = np.hstack([cv2.resize(fit(t), (w // 2, h // 2)) for t in list(targets)[:4]])
    top = cv2.resize(top, (2 * w, h // 2))
    cv2.putText(top, "goal views: front | left | back | right", (4, 12), font, 0.4, (0, 255, 255), 1)
    rows = [top]
    for i, c in enumerate(cands):
        goal = fit(targets[c.view]) if c.view is not None and c.view < len(targets) else np.zeros((h, w, 3), np.uint8)
        row = np.hstack([fit(image_of(c.node)), goal])
        colour = (0, 255, 0) if i == chosen else (0, 255, 255)
        cv2.putText(row, c.label(i) + ("  <- current" if i == chosen else ""), (4, 14), font, 0.38, colour, 1)
        cv2.putText(row, f"goal {VIEW_NAMES[c.view] if c.view is not None and c.view < 4 else '?'}",
                    (w + 4, 14), font, 0.4, (0, 255, 255), 1)
        rows.append(row)
    foot = np.zeros((40, 2 * w, 3), np.uint8)
    cv2.putText(foot, "Same panels in the same order beside the poster = real place.", (4, 15), font, 0.4, (255, 255, 255), 1)
    cv2.putText(foot, "One matching poster with different neighbours = look-alike.", (4, 32), font, 0.4, (255, 255, 255), 1)
    rows.append(foot)
    return np.vstack(rows)


def as_dict(c: Candidate) -> dict:
    return asdict(c)
