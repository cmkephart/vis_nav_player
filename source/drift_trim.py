"""
Forward-drift trim. On the course server, driving forward also turns the robot (trial 515b23a3c14a: about
+1.2..+1.7 deg LEFT per forward tick, growing through the drive; local mazes: 0). We measure it from consecutive
camera frames on the driver's forward ticks and cancel it by turning some plain FORWARD ticks into FORWARD+turn
(still one tick each, so it costs nothing in SPL). Off until the estimate is consistent.
"""

from __future__ import annotations

from collections import deque

import numpy as np

TURN_DEG = 3.44           # deg per turn tick (measured: 3.40-3.44 on the server and locally)
WINDOW = 40               # forward-tick samples kept
MIN_SAMPLES = 30
MIN_DRIFT = 0.6           # deg/tick: below this, no trim (server trial: 1.2-2.4; local: 0; 0.4 let a local drive trim 12x)
MIN_AGREE = 0.7           # fraction of samples on the drift's side


class DriftTrim:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.samples: deque[float] = deque(maxlen=WINDOW)
        self.acc = 0.0             # deg of drift not yet cancelled (left +)
        self.trims = 0

    def observe(self, dyaw: float | None, sent: str, count: int) -> None:
        """dyaw: measured turn (deg, left +) between the frames before and after a 1-tick action `sent`."""
        if dyaw is None or count != 1 or "FORWARD" not in sent or "BACKWARD" in sent:
            return
        keyed = (TURN_DEG if "LEFT" in sent else 0.0) - (TURN_DEG if "RIGHT" in sent else 0.0)
        self.samples.append(dyaw - keyed)

    def drift(self) -> float | None:
        """deg per forward tick (left +), or None while not confident."""
        if len(self.samples) < MIN_SAMPLES:
            return None
        v = np.array(self.samples)
        d = float(np.median(v))
        if abs(d) < MIN_DRIFT or float(np.mean(np.sign(v) == np.sign(d))) < MIN_AGREE:
            return None
        return d

    def correct(self, wants_left: bool, wants_right: bool) -> str | None:
        """For a plain FORWARD tick: 'LEFT'/'RIGHT' to add to it, or None. Call once per plain forward tick."""
        d = self.drift()
        if not self.enabled or d is None or wants_left or wants_right:
            return None
        self.acc += d
        if self.acc >= TURN_DEG / 2:
            self.acc -= TURN_DEG
            self.trims += 1
            return "RIGHT"
        if self.acc <= -TURN_DEG / 2:
            self.acc += TURN_DEG
            self.trims += 1
            return "LEFT"
        return None

    def status(self) -> str:
        d = self.drift()
        n = len(self.samples)
        if d is None:
            return f"drift: measuring ({n} samples)" if n < MIN_SAMPLES else "drift: none"
        side = "LEFT" if d > 0 else "RIGHT"
        return f"drift: forward veers {side} {abs(d):.1f} deg/tick, trim {'ON' if self.enabled else 'OFF'} ({self.trims})"
