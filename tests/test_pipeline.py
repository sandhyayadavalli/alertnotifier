"""Smoke test for the shared end-to-end pipeline: it runs, scores every event,
and scores flagged events higher than the rest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.pipeline import PipelineResult, run_pipeline  # noqa: E402


def _events(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        bad = i % 12 == 0  # every 12th event is blatantly anomalous
        rows.append({
            "event_id": f"e{i}", "entity_id": f"CLI-{i % 5:04d}",
            "timestamp": pd.Timestamp("2026-06-01") + pd.Timedelta(minutes=i),
            "latency_ms": float(3000 if bad else rng.normal(100, 20)),
            "payload_bytes": int(rng.normal(1500, 300)),
            "req_count_5m": int(20 if bad else 1),
            "distinct_endpoints_5m": int(7 if bad else 1),
            "minutes_since_last_request": float(rng.uniform(1, 20)),
            "is_error": bool(bad), "new_ip": bool(bad), "always_on": False,
            "off_baseline_hour": False, "client_tier": "standard", "base_client_tier": "standard",
            "severity_hint": "high" if bad else "low",
        })
    return pd.DataFrame(rows)


def test_pipeline_runs_end_to_end():
    result = run_pipeline(_events())
    assert isinstance(result, PipelineResult)
    assert len(result.risk) == 60          # one score per event
    assert isinstance(result.incidents, list)
    assert set(["event_id", "entity_id", "risk_score", "signals"]).issubset(result.alerts.columns)


def test_flagged_events_score_higher():
    result = run_pipeline(_events())
    flagged = result.fired.any(axis=1)
    assert flagged.sum() >= 1
    assert result.risk[flagged].mean() > result.risk[~flagged].mean()
