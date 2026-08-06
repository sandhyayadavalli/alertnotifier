"""Incident correlation groups related alerts, and the escalation matrix routes
by score. These are the checks that separate a monitoring *system* from a
detector that just fires N times."""
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.correlation.incidents import Incident, correlate  # noqa: E402
from alertnotifier.escalation.matrix import EscalationMatrix  # noqa: E402

BASE = pd.Timestamp("2026-06-01 00:00")


def _alerts(rows):
    return pd.DataFrame(rows)


def test_close_alerts_merge_into_one_incident():
    inc = correlate(_alerts([
        {"entity_id": "c1", "timestamp": BASE, "risk_score": 60, "signals": ["rate_anomaly"]},
        {"entity_id": "c1", "timestamp": BASE + timedelta(minutes=5), "risk_score": 80, "signals": ["isolation_forest"]},
    ]), gap_minutes=30)
    assert len(inc) == 1
    assert inc[0].alert_count == 2
    assert inc[0].score == 80  # consolidated = max
    assert set(inc[0].signals) == {"rate_anomaly", "isolation_forest"}


def test_alerts_far_apart_split_into_two():
    inc = correlate(_alerts([
        {"entity_id": "c1", "timestamp": BASE, "risk_score": 60, "signals": ["rate_anomaly"]},
        {"entity_id": "c1", "timestamp": BASE + timedelta(hours=2), "risk_score": 50, "signals": ["off_baseline"]},
    ]), gap_minutes=30)
    assert len(inc) == 2


def test_different_entities_are_separate_incidents():
    inc = correlate(_alerts([
        {"entity_id": "c1", "timestamp": BASE, "risk_score": 60, "signals": ["rate_anomaly"]},
        {"entity_id": "c2", "timestamp": BASE, "risk_score": 50, "signals": ["off_baseline"]},
    ]), gap_minutes=30)
    assert len(inc) == 2


def test_escalation_bands_route_by_score():
    m = EscalationMatrix({
        "bands": [
            {"min": 0, "max": 25, "action": "log_only"},
            {"min": 25, "max": 50, "action": "notify_analyst"},
            {"min": 50, "max": 75, "action": "notify_security", "sla_minutes": 60},
            {"min": 75, "max": 101, "action": "auto_contain"},
        ],
        "actions": {},
    })
    assert m.route(10)["action"] == "log_only"
    assert m.route(30)["action"] == "notify_analyst"
    assert m.route(60)["action"] == "notify_security"
    assert m.route(95)["action"] == "auto_contain"


def test_apply_sets_action_and_lifecycle():
    inc = Incident("INC-1", "c1", BASE, BASE, 3, ["isolation_forest"], 90.0)
    EscalationMatrix.from_yaml().apply(inc)
    assert inc.action == "auto_contain"
    assert inc.status == "escalated"
