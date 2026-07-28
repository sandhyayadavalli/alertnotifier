"""Replay harness — stream the engine's decisions in chronological order.

Detection/scoring/correlation are computed up front (batch); this then replays
the resulting incidents in time order, optionally pacing by wall-clock (a speed
factor) so the pipeline "runs live". It's the stream that feeds the audit log
now and the Phase 7 dashboard later.
"""
from __future__ import annotations

import time
from typing import Iterator

from alertnotifier.correlation.incidents import Incident


def replay(incidents: list[Incident], speed: float = 0.0) -> Iterator[Incident]:
    """Yield incidents ordered by start time.

    If ``speed`` > 0, sleep between incidents by (real gap / speed), capped at
    2s, to simulate a live feed. ``speed`` == 0 replays as fast as possible.
    """
    prev_start = None
    for inc in sorted(incidents, key=lambda i: i.start):
        if speed > 0 and prev_start is not None:
            gap = (inc.start - prev_start).total_seconds()
            time.sleep(min(max(gap, 0.0) / speed, 2.0))
        prev_start = inc.start
        yield inc
