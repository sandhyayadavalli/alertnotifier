#!/usr/bin/env python
"""Replay the engine's incidents into a hash-chained audit log, then prove the
log is tamper-evident by editing a past row and re-verifying.

Run:  python scripts/run_replay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.audit.log import AuditLog  # noqa: E402
from alertnotifier.correlation.incidents import correlate  # noqa: E402
from alertnotifier.dataset import load_dataset, split_events_labels  # noqa: E402
from alertnotifier.detectors.ml_anomaly import IsolationForestDetector  # noqa: E402
from alertnotifier.detectors.rules import default_detectors  # noqa: E402
from alertnotifier.escalation.matrix import EscalationMatrix  # noqa: E402
from alertnotifier.ingestion.replay import replay  # noqa: E402
from alertnotifier.scoring.engine import RiskScorer  # noqa: E402


def _build_incidents() -> list:
    df = load_dataset("api_events")
    events, _ = split_events_labels(df)
    fired = {d.name: d.detect(events).reindex(events["event_id"]).fillna(False) for d in default_detectors()}
    iso = IsolationForestDetector(contamination=0.10).fit(events)
    fired["isolation_forest"] = iso.detect(events).reindex(events["event_id"]).fillna(False)
    fired = pd.DataFrame(fired)
    risk = RiskScorer.from_yaml().score(events, fired)

    flagged = fired.any(axis=1)
    alert_ids = flagged[flagged].index
    ev = events.set_index("event_id")
    signals = fired.loc[alert_ids].apply(lambda r: [c for c in fired.columns if r[c]], axis=1)
    alerts = pd.DataFrame({
        "entity_id": ev.loc[alert_ids, "entity_id"].to_numpy(),
        "timestamp": ev.loc[alert_ids, "timestamp"].to_numpy(),
        "risk_score": risk.reindex(alert_ids).to_numpy(),
        "signals": signals.to_numpy(),
    })
    incidents = correlate(alerts, gap_minutes=30.0)
    matrix = EscalationMatrix.from_yaml()
    for inc in incidents:
        matrix.apply(inc)
    return incidents


def main() -> None:
    incidents = _build_incidents()

    db = Path(__file__).resolve().parents[1] / "data" / "generated" / "audit.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    audit = AuditLog(str(db).replace("\\", "/"))

    # replay the incidents (time-ordered) into the audit log
    audit.extend({
        "incident_id": inc.incident_id, "entity_id": inc.entity_id,
        "start_ts": inc.start.isoformat(), "end_ts": inc.end.isoformat(),
        "alert_count": inc.alert_count, "detectors": ",".join(inc.signals),
        "risk_score": inc.score, "action": inc.action,
    } for inc in replay(incidents, speed=0.0))

    ok, bad = audit.verify()
    print(f"Replayed {audit.count()} incidents into a hash-chained audit log (SQLite).")
    print(f"  chain intact: {ok}")

    print("\nSample rows (each row_hash is chained from the previous):")
    print(f"  {'seq':>4}  {'incident':10s} {'score':>5s}  {'action':16s} {'prev_hash':10s} {'row_hash':10s}")
    for r in audit.entries(limit=5):
        print(f"  {r.seq:>4}  {r.incident_id:10s} {r.risk_score:5.1f}  {r.action:16s} {r.prev_hash[:8]}.. {r.row_hash[:8]}..")

    # --- tamper-evidence demo: an attacker edits a past row directly in the DB ---
    target = max(1, audit.count() // 2)
    with audit.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_log SET risk_score = 1.0, action = 'log_only' WHERE seq = :s"),
            {"s": target},
        )
    ok2, bad2 = audit.verify()
    print(f"\nSimulated tampering: downgrade the incident at seq {target} (score->1.0, action->log_only)")
    print(f"  chain intact: {ok2}   first broken row: seq {bad2}")
    print("  -> detected: the edited row's hash no longer matches the chain.")


if __name__ == "__main__":
    main()
