"""Rule-based detectors for the generic anomaly taxonomy.

Each detector reads the *event view* only (never the ``gt_*`` columns) and
returns a boolean Series indexed by ``event_id`` — True where the rule fires.
Detectors use per-client baseline features where available; that's what
separates a principled rule from a naive global threshold.

``multivariate_outlier`` deliberately has NO rule here — it is held out for the
ML layer (Phase 3), so the evaluation shows exactly the gap the model must
close.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def _robust_z(s: pd.Series) -> pd.Series:
    """Median/MAD z-score — robust to the very outliers we're hunting."""
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else 1e-9
    return (s - med) / scale


class Detector(ABC):
    #: human-readable name
    name: str
    #: the gt_anomaly_type this rule is designed to catch
    target: str

    @abstractmethod
    def detect(self, events: pd.DataFrame) -> pd.Series:
        """Return a boolean Series indexed by ``event_id``."""

    def _as_series(self, events: pd.DataFrame, fired) -> pd.Series:
        return pd.Series(
            np.asarray(fired, dtype=bool), index=events["event_id"].to_numpy(), name=self.name
        )


class ValueSpikeDetector(Detector):
    """Latency/payload that is extreme *relative to the client's own baseline*
    (per-client robust z-score), so a normally-slow client isn't over-flagged."""

    name = "value_spike"
    target = "value_spike"

    def __init__(self, z: float = 6.0):
        self.z = z

    def detect(self, events: pd.DataFrame) -> pd.Series:
        g = events.groupby("entity_id")
        lat_z = g["latency_ms"].transform(_robust_z)
        pay_z = g["payload_bytes"].transform(_robust_z)
        return self._as_series(events, ((lat_z > self.z) | (pay_z > self.z)).to_numpy())


class SustainedDriftDetector(Detector):
    """A rolling per-client error rate that stays high over a window — a broken
    integration — as opposed to isolated one-off errors (normal noise)."""

    name = "sustained_drift"
    target = "sustained_drift"

    def __init__(self, window: int = 8, min_error_rate: float = 0.5):
        self.window = window
        self.min_error_rate = min_error_rate

    def detect(self, events: pd.DataFrame) -> pd.Series:
        ev = events.sort_values(["entity_id", "timestamp"])
        roll = ev.groupby("entity_id")["is_error"].transform(
            lambda s: s.astype(float).rolling(self.window, min_periods=self.window).mean()
        )
        return self._as_series(ev, (roll.fillna(0.0) >= self.min_error_rate).to_numpy())


class RateAnomalyDetector(Detector):
    """A request-rate burst, OR a client going silent. A long gap only counts as
    silence for a client expected to be *always on* — a naive global gap rule
    would false-positive on business clients overnight."""

    name = "rate_anomaly"
    target = "rate_anomaly"

    def __init__(self, burst_count: int = 8, silence_minutes: float = 180.0):
        self.burst_count = burst_count
        self.silence_minutes = silence_minutes

    def detect(self, events: pd.DataFrame) -> pd.Series:
        burst = events["req_count_5m"] > self.burst_count
        silence = events["always_on"] & (events["minutes_since_last_request"] > self.silence_minutes)
        return self._as_series(events, (burst | silence).to_numpy())


class OffBaselineDetector(Detector):
    """Activity at an hour outside THIS client's own active window. The
    per-client flag is the point: a global night-hour rule over-fires on
    always-on clients."""

    name = "off_baseline"
    target = "off_baseline"

    def detect(self, events: pd.DataFrame) -> pd.Series:
        return self._as_series(events, events["off_baseline_hour"].to_numpy(dtype=bool))


class StatusChangeDetector(Detector):
    """The client's access tier differs from its baseline tier."""

    name = "status_change"
    target = "status_change"

    def detect(self, events: pd.DataFrame) -> pd.Series:
        return self._as_series(events, (events["client_tier"] != events["base_client_tier"]).to_numpy())


def default_detectors() -> list[Detector]:
    return [
        ValueSpikeDetector(),
        SustainedDriftDetector(),
        RateAnomalyDetector(),
        OffBaselineDetector(),
        StatusChangeDetector(),
    ]
