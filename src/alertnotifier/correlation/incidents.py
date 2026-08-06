"""Incident correlation — group raw alerts into a handful of incidents.

The operational point of a monitoring system: a detector that fires thousands of
times is useless. Here, alerts from the *same entity* that occur close together
in time are merged into a single incident (sessionization by a time gap). Each
incident carries a consolidated risk score (the max of its alerts), the set of
signals involved, its duration, and a lifecycle status. Escalation (see
``escalation/matrix.py``) then acts on incidents, not raw alerts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd


@dataclass
class Incident:
    incident_id: str
    entity_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    alert_count: int
    signals: list[str]
    score: float
    status: str = "open"
    action: str = ""
    sla_minutes: Optional[int] = None

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0


def correlate(alerts: pd.DataFrame, gap_minutes: float = 30.0) -> list[Incident]:
    """Group alerts into incidents.

    ``alerts`` needs columns: entity_id, timestamp, risk_score, signals
    (a list of the signal names that fired for that event). Alerts from the same
    entity within ``gap_minutes`` of each other belong to one incident.
    """
    incidents: list[Incident] = []
    if alerts.empty:
        return incidents

    gap = timedelta(minutes=gap_minutes)
    ordered = alerts.sort_values(["entity_id", "timestamp"])
    seq = 0
    for entity, grp in ordered.groupby("entity_id", sort=False):
        current: Optional[Incident] = None
        prev_ts: Optional[pd.Timestamp] = None
        for row in grp.itertuples(index=False):
            ts = pd.Timestamp(row.timestamp)
            if current is None or (ts - prev_ts) > gap:
                if current is not None:
                    incidents.append(current)
                seq += 1
                current = Incident(
                    incident_id=f"INC-{seq:05d}", entity_id=str(entity),
                    start=ts, end=ts, alert_count=1,
                    signals=list(dict.fromkeys(row.signals)), score=float(row.risk_score),
                )
            else:
                current.end = ts
                current.alert_count += 1
                for s in row.signals:
                    if s not in current.signals:
                        current.signals.append(s)
                current.score = max(current.score, float(row.risk_score))
            prev_ts = ts
        if current is not None:
            incidents.append(current)
    return incidents
