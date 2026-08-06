"""End-to-end pipeline assembly: events → detections → scores → incidents.

One place that wires the rule detectors, the IsolationForest, scoring,
correlation, and escalation together and returns a structured result — shared by
the dashboard (and available to the scripts) so the wiring lives in one spot.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alertnotifier.correlation.incidents import Incident, correlate
from alertnotifier.detectors.ml_anomaly import IsolationForestDetector
from alertnotifier.detectors.rules import default_detectors
from alertnotifier.escalation.matrix import EscalationMatrix
from alertnotifier.scoring.engine import RiskScorer


@dataclass
class PipelineResult:
    events: pd.DataFrame          # the event view (no labels)
    fired: pd.DataFrame           # per-event signal flags (bool), indexed by event_id
    risk: pd.Series               # per-event risk score 0-100, indexed by event_id
    alerts: pd.DataFrame          # flagged events: event_id, entity_id, timestamp, risk_score, signals
    incidents: list[Incident]     # correlated + escalated


def run_pipeline(events: pd.DataFrame, contamination: float = 0.10, gap_minutes: float = 30.0) -> PipelineResult:
    # detection: rules + ML
    fired = {d.name: d.detect(events).reindex(events["event_id"]).fillna(False) for d in default_detectors()}
    iso = IsolationForestDetector(contamination=contamination).fit(events)
    fired["isolation_forest"] = iso.detect(events).reindex(events["event_id"]).fillna(False)
    fired = pd.DataFrame(fired)

    # scoring
    risk = RiskScorer.from_yaml().score(events, fired)

    # alerts = any signal fired, carrying the score + which signals
    flagged = fired.any(axis=1)
    alert_ids = flagged[flagged].index
    ev = events.set_index("event_id")
    signals = fired.loc[alert_ids].apply(lambda r: [c for c in fired.columns if r[c]], axis=1)
    alerts = pd.DataFrame({
        "event_id": list(alert_ids),
        "entity_id": ev.loc[alert_ids, "entity_id"].to_numpy(),
        "timestamp": ev.loc[alert_ids, "timestamp"].to_numpy(),
        "risk_score": risk.reindex(alert_ids).to_numpy(),
        "signals": signals.to_numpy(),
    })

    # correlation + escalation
    incidents = correlate(alerts, gap_minutes=gap_minutes)
    matrix = EscalationMatrix.from_yaml()
    for inc in incidents:
        matrix.apply(inc)

    return PipelineResult(events=events, fired=fired, risk=risk, alerts=alerts, incidents=incidents)
