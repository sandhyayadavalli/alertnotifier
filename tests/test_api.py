"""The FastAPI ingestion endpoint validates events and returns a live risk score
+ escalation action. A fast in-memory engine is injected so the test doesn't load
the full dataset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient  # noqa: E402

from alertnotifier.ingestion import api  # noqa: E402


def _train(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        bad = i % 12 == 0
        rows.append({
            "event_id": f"e{i}", "entity_id": f"CLI-{i % 5:04d}",
            "timestamp": pd.Timestamp("2026-06-01") + pd.Timedelta(minutes=i),
            "latency_ms": float(3000 if bad else rng.normal(100, 20)),
            "payload_bytes": int(rng.normal(1500, 300)), "req_count_5m": int(20 if bad else 1),
            "distinct_endpoints_5m": int(7 if bad else 1),
            "minutes_since_last_request": float(rng.uniform(1, 20)),
            "is_error": bool(bad), "new_ip": bool(bad), "always_on": False,
            "off_baseline_hour": False, "client_tier": "standard", "base_client_tier": "standard",
            "severity_hint": "high" if bad else "low",
        })
    return pd.DataFrame(rows)


def _client() -> TestClient:
    api.set_engine(api.ScoringEngine(_train()))
    return TestClient(api.app)


def test_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_post_event_scores_and_routes():
    body = [{
        "event_id": "live-1", "timestamp": "2026-06-08T03:00:00Z", "entity_id": "CLI-0001",
        "entity_type": "api_client", "source": "api", "event_type": "api_request",
        "severity_hint": "high",
        "attributes": {
            "latency_ms": 50.0, "payload_bytes": 300, "req_count_5m": 25, "distinct_endpoints_5m": 8,
            "minutes_since_last_request": 2.0, "is_error": False, "new_ip": True, "always_on": False,
            "off_baseline_hour": True, "client_tier": "standard", "base_client_tier": "standard",
        },
    }]
    r = _client().post("/events", json=body)
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["event_id"] == "live-1"
    assert 0 <= result["risk_score"] <= 100
    assert result["action"] in {"log_only", "notify_analyst", "notify_security", "auto_contain"}
    assert result["signals"]  # a burst on an off-baseline hour from a new IP should fire something


def test_rejects_malformed_event():
    r = _client().post("/events", json=[{"entity_id": "CLI-0001"}])  # missing required fields
    assert r.status_code == 422
