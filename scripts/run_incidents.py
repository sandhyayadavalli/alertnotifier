#!/usr/bin/env python
"""Correlate raw alerts into incidents and route them through the escalation
matrix — the step that makes this a monitoring *system*, not just a detector.

Run:  python scripts/run_incidents.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.correlation.incidents import correlate  # noqa: E402
from alertnotifier.dataset import load_dataset, split_events_labels  # noqa: E402
from alertnotifier.detectors.ml_anomaly import IsolationForestDetector  # noqa: E402
from alertnotifier.detectors.rules import default_detectors  # noqa: E402
from alertnotifier.escalation.matrix import EscalationMatrix  # noqa: E402
from alertnotifier.scoring.engine import RiskScorer  # noqa: E402


def main() -> None:
    df = load_dataset("api_events")
    events, _ = split_events_labels(df)

    # per-event signals (rules + ML) and risk score
    fired = {d.name: d.detect(events).reindex(events["event_id"]).fillna(False) for d in default_detectors()}
    iso = IsolationForestDetector(contamination=0.10).fit(events)
    fired["isolation_forest"] = iso.detect(events).reindex(events["event_id"]).fillna(False)
    fired = pd.DataFrame(fired)
    risk = RiskScorer.from_yaml().score(events, fired)

    # an "alert" = any signal fired
    flagged = fired.any(axis=1)
    alert_ids = flagged[flagged].index
    ev = events.set_index("event_id")
    signals_per = fired.loc[alert_ids].apply(lambda r: [c for c in fired.columns if r[c]], axis=1)
    alerts = pd.DataFrame({
        "entity_id": ev.loc[alert_ids, "entity_id"].to_numpy(),
        "timestamp": ev.loc[alert_ids, "timestamp"].to_numpy(),
        "risk_score": risk.reindex(alert_ids).to_numpy(),
        "signals": signals_per.to_numpy(),
    })

    incidents = correlate(alerts, gap_minutes=30.0)
    matrix = EscalationMatrix.from_yaml()
    for inc in incidents:
        matrix.apply(inc)

    n_events, n_alerts, n_inc = len(events), len(alerts), len(incidents)
    print(f"Consolidation:  {n_events} events  ->  {n_alerts} raw alerts  ->  {n_inc} incidents")
    print(f"  a human reviews {n_inc} incidents instead of {n_alerts} alerts "
          f"(~{n_alerts / max(1, n_inc):.0f}x fewer)")

    escalated = sum(1 for i in incidents if i.status == "escalated")
    print(f"\nLifecycle:  {escalated} escalated, {n_inc - escalated} auto-resolved (log-only)")

    by_action = Counter(inc.action for inc in incidents)
    print("\nIncidents by escalation action:")
    for action in ["auto_contain", "notify_security", "notify_analyst", "log_only"]:
        if by_action.get(action):
            print(f"  {action:16s} {by_action[action]:4d}   ({matrix.action_description(action)})")

    print("\nTop incidents by risk score:")
    print(f"  {'incident':10s} {'client':9s} {'score':>5s} {'alerts':>6s} {'min':>5s} {'action':16s} signals")
    for inc in sorted(incidents, key=lambda i: i.score, reverse=True)[:8]:
        print(f"  {inc.incident_id:10s} {inc.entity_id:9s} {inc.score:5.1f} {inc.alert_count:6d} "
              f"{inc.duration_minutes:5.0f} {inc.action:16s} {','.join(inc.signals)}")


if __name__ == "__main__":
    main()
