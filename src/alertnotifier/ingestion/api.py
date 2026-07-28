"""FastAPI ingestion endpoint — `POST /events` scores events live.

Models are trained offline, served online: the IsolationForest is fit once on
the historical dataset at first use. Each posted event is validated against the
`Event` schema, then scored by the ML model plus the *stateless* rule detectors
(the ones that read enrichment attributes rather than per-entity history), given
a 0-100 risk score, and routed through the escalation matrix. The stateful
detectors (per-client value spikes, sustained drift) need entity history and run
in the batch pipeline, not on a lone event.

Run:  uvicorn --app-dir src alertnotifier.ingestion.api:app --reload
Then: POST a JSON list of events to http://127.0.0.1:8000/events
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alertnotifier.dataset import load_dataset, split_events_labels  # noqa: E402
from alertnotifier.detectors.ml_anomaly import IsolationForestDetector  # noqa: E402
from alertnotifier.detectors.rules import (  # noqa: E402
    OffBaselineDetector, RateAnomalyDetector, StatusChangeDetector,
)
from alertnotifier.escalation.matrix import EscalationMatrix  # noqa: E402
from alertnotifier.models import Event  # noqa: E402
from alertnotifier.scoring.engine import RiskScorer  # noqa: E402


class ScoringEngine:
    """A fit ML model + stateless detectors + scorer + escalation matrix."""

    def __init__(self, train_events: pd.DataFrame):
        self.iso = IsolationForestDetector(contamination=0.10).fit(train_events)
        self.rules = [RateAnomalyDetector(), OffBaselineDetector(), StatusChangeDetector()]
        self.scorer = RiskScorer.from_yaml()
        self.matrix = EscalationMatrix.from_yaml()

    def score(self, df: pd.DataFrame) -> list[dict]:
        fired = {d.name: d.detect(df).reindex(df["event_id"]).fillna(False) for d in self.rules}
        fired["isolation_forest"] = self.iso.detect(df).reindex(df["event_id"]).fillna(False)
        fired = pd.DataFrame(fired)
        risk = self.scorer.score(df, fired)
        dfi = df.set_index("event_id")
        out = []
        for eid in df["event_id"]:
            score = float(risk.loc[eid])
            out.append({
                "event_id": eid,
                "entity_id": str(dfi.loc[eid, "entity_id"]),
                "risk_score": round(score, 1),
                "signals": [c for c in fired.columns if bool(fired.loc[eid, c])],
                "action": self.matrix.route(score)["action"],
            })
        return out


def _to_frame(events: list[Event]) -> pd.DataFrame:
    return pd.DataFrame([{
        "event_id": ev.event_id, "timestamp": pd.Timestamp(ev.timestamp), "entity_id": ev.entity_id,
        "entity_type": ev.entity_type, "source": ev.source, "event_type": ev.event_type,
        "severity_hint": ev.severity_hint.value, **ev.attributes,
    } for ev in events])


app = FastAPI(title="AlertNotifier", description="Live risk scoring for entity events.")
_engine: Optional[ScoringEngine] = None


def get_engine() -> ScoringEngine:
    global _engine
    if _engine is None:
        events, _ = split_events_labels(load_dataset("api_events"))
        _engine = ScoringEngine(events)
    return _engine


def set_engine(engine: ScoringEngine) -> None:
    """Inject a pre-built engine (used by tests to avoid loading the full dataset)."""
    global _engine
    _engine = engine


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/events")
def post_events(events: list[Event]) -> dict:
    """Validate + score a batch of events; returns per-event risk + escalation action."""
    results = get_engine().score(_to_frame(events))
    return {"scored": len(results), "results": results}
