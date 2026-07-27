"""Risk-scoring engine: score is bounded, normal stays low, and each component
(detection severity, entity criticality, persistence/recurrence) pushes it up."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.scoring.engine import RiskScorer  # noqa: E402

CFG = {
    "weights": {"detection": 0.55, "severity_hint": 0.10, "persistence": 0.20, "recurrence": 0.15},
    "detector_severity": {"value_spike": 0.5, "status_change": 0.7, "isolation_forest": 0.8},
    "severity_hint": {"low": 0.1, "medium": 0.4, "high": 0.7},
    "criticality": {"throttled": 0.8, "standard": 1.0, "trusted": 1.3},
    "persistence_window_events": 10,
    "recurrence_cap": 20,
}
COLS = ["value_spike", "status_change", "isolation_forest"]


def _fired(index, fired_value_spike):
    return pd.DataFrame(
        {"value_spike": fired_value_spike, "status_change": [False] * len(index),
         "isolation_forest": [False] * len(index)},
        index=index,
    )


def test_normal_event_scores_low_and_bounded():
    events = pd.DataFrame([{
        "event_id": "n", "entity_id": "c1", "timestamp": pd.Timestamp("2026-06-01"),
        "severity_hint": "low", "base_client_tier": "standard",
    }])
    s = RiskScorer(CFG).score(events, _fired(["n"], [False]))
    assert 0 <= s["n"] <= 100
    assert s["n"] < 10


def test_anomaly_scores_higher_than_normal():
    events = pd.DataFrame([
        {"event_id": "a", "entity_id": "c1", "timestamp": pd.Timestamp("2026-06-01"),
         "severity_hint": "high", "base_client_tier": "trusted"},
        {"event_id": "n", "entity_id": "c2", "timestamp": pd.Timestamp("2026-06-01"),
         "severity_hint": "low", "base_client_tier": "standard"},
    ])
    fired = pd.DataFrame(
        {"value_spike": [False, False], "status_change": [True, False],
         "isolation_forest": [False, False]}, index=["a", "n"])
    s = RiskScorer(CFG).score(events, fired)
    assert s["a"] > s["n"]
    assert s["a"] > 40


def test_criticality_raises_score():
    events = pd.DataFrame([
        {"event_id": "t", "entity_id": "c1", "timestamp": pd.Timestamp("2026-06-01"),
         "severity_hint": "low", "base_client_tier": "trusted"},
        {"event_id": "h", "entity_id": "c2", "timestamp": pd.Timestamp("2026-06-01"),
         "severity_hint": "low", "base_client_tier": "throttled"},
    ])
    s = RiskScorer(CFG).score(events, _fired(["t", "h"], [True, True]))
    assert s["t"] > s["h"]  # same detection, trusted client scores higher


def test_persistence_and_recurrence_raise_score():
    rows, vs = [], []
    for i in range(10):  # c1: sustained — every event flagged
        rows.append({"event_id": f"a{i}", "entity_id": "c1",
                     "timestamp": pd.Timestamp("2026-06-01") + pd.Timedelta(minutes=i),
                     "severity_hint": "low", "base_client_tier": "standard"})
        vs.append(True)
    for i in range(10):  # c2: one-off — only the last event flagged
        rows.append({"event_id": f"b{i}", "entity_id": "c2",
                     "timestamp": pd.Timestamp("2026-06-01") + pd.Timedelta(minutes=i),
                     "severity_hint": "low", "base_client_tier": "standard"})
        vs.append(i == 9)
    events = pd.DataFrame(rows)
    s = RiskScorer(CFG).score(events, _fired(events["event_id"], vs))
    assert s["a9"] > s["b9"]  # sustained + recurring beats a one-off, same detection
