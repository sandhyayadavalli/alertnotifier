"""Hand-crafted edge cases for the rule detectors.

The interesting cases are the ones that separate a principled per-client rule
from a naive global one — e.g. an always-slow client is not a spike, and a
business client's overnight gap is not "silence".
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.detectors.rules import (  # noqa: E402
    OffBaselineDetector,
    RateAnomalyDetector,
    StatusChangeDetector,
    SustainedDriftDetector,
    ValueSpikeDetector,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_value_spike_fires_on_per_client_outlier():
    rows = [
        {"event_id": f"e{i}", "entity_id": "c1", "latency_ms": 100, "payload_bytes": 1000}
        for i in range(20)
    ]
    rows.append({"event_id": "spike", "entity_id": "c1", "latency_ms": 5000, "payload_bytes": 1000})
    out = ValueSpikeDetector().detect(_df(rows))
    assert bool(out["spike"]) is True
    assert int(out.drop("spike").sum()) == 0


def test_value_spike_ignores_uniformly_slow_client():
    # a client that is ALWAYS slow is not anomalous relative to itself
    rows = [
        {"event_id": f"e{i}", "entity_id": "c1", "latency_ms": 5000, "payload_bytes": 1000}
        for i in range(20)
    ]
    out = ValueSpikeDetector().detect(_df(rows))
    assert int(out.sum()) == 0


def test_sustained_drift_fires_on_run_not_scattered_errors():
    rows = [
        {"event_id": f"n{i}", "entity_id": "c1", "timestamp": i, "is_error": (i % 10 == 0)}
        for i in range(30)
    ]
    rows += [
        {"event_id": f"d{j}", "entity_id": "c1", "timestamp": 100 + j, "is_error": True}
        for j in range(10)
    ]
    out = SustainedDriftDetector(window=8, min_error_rate=0.5).detect(_df(rows))
    assert int(out.filter(like="d").sum()) >= 3   # sustained run caught
    assert int(out.filter(like="n").sum()) == 0   # scattered errors are noise


def test_rate_anomaly_burst():
    rows = [{"event_id": "b", "entity_id": "c1", "req_count_5m": 20,
             "always_on": False, "minutes_since_last_request": 1}]
    assert bool(RateAnomalyDetector().detect(_df(rows))["b"]) is True


def test_rate_anomaly_silence_only_for_always_on():
    rows = [
        {"event_id": "on", "entity_id": "c1", "req_count_5m": 1,
         "always_on": True, "minutes_since_last_request": 600},
        {"event_id": "biz", "entity_id": "c2", "req_count_5m": 1,
         "always_on": False, "minutes_since_last_request": 600},
    ]
    out = RateAnomalyDetector(silence_minutes=180).detect(_df(rows))
    assert bool(out["on"]) is True    # always-on gap = silence
    assert bool(out["biz"]) is False  # business overnight gap = normal


def test_off_baseline_uses_per_client_flag():
    rows = [
        {"event_id": "a", "entity_id": "c1", "off_baseline_hour": True},
        {"event_id": "b", "entity_id": "c1", "off_baseline_hour": False},
    ]
    out = OffBaselineDetector().detect(_df(rows))
    assert bool(out["a"]) is True and bool(out["b"]) is False


def test_status_change():
    rows = [
        {"event_id": "a", "entity_id": "c1", "client_tier": "trusted", "base_client_tier": "standard"},
        {"event_id": "b", "entity_id": "c1", "client_tier": "standard", "base_client_tier": "standard"},
    ]
    out = StatusChangeDetector().detect(_df(rows))
    assert bool(out["a"]) is True and bool(out["b"]) is False
