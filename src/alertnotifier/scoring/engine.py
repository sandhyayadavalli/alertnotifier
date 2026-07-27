"""Config-driven risk scoring — turns raw detections into a single 0-100 score.

The score is a weighted blend of four components, each in [0, 1]:

* **detection**  — the severity of the strongest signal that fired (rules + ML)
* **severity_hint** — the source system's own coarse hint
* **persistence** — the fraction of the entity's recent events that were flagged
  (a sustained attack scores higher than a one-off blip)
* **recurrence**  — how many times the entity has been flagged over the whole
  window (a repeat offender scores higher)

The weighted blend is then scaled by an **entity-criticality** multiplier (a more
privileged client is more dangerous) and clipped to 0-100. All weights and maps
live in ``config/scoring_config.yaml`` so behaviour is tunable without code.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "scoring_config.yaml"


class RiskScorer:
    def __init__(self, config: dict):
        self.cfg = config

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "RiskScorer":
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def score(self, events: pd.DataFrame, fired: pd.DataFrame) -> pd.Series:
        """Return a 0-100 risk score per event.

        ``events`` needs columns: event_id, entity_id, timestamp, severity_hint,
        base_client_tier. ``fired`` is a boolean DataFrame indexed by event_id
        with one column per signal name (rule detectors + ``isolation_forest``).
        """
        ev = events.set_index("event_id")
        fired = fired.reindex(ev.index).fillna(False).astype(bool)

        # detection = severity of the strongest signal that fired
        sev = pd.Series(self.cfg["detector_severity"]).reindex(fired.columns).fillna(0.0)
        detection = fired.mul(sev, axis=1).max(axis=1).fillna(0.0)
        flagged = fired.any(axis=1)

        hint = ev["severity_hint"].map(self.cfg["severity_hint"]).fillna(0.0)
        crit = ev["base_client_tier"].map(self.cfg["criticality"]).fillna(1.0)

        # persistence & recurrence need per-entity time order
        order = ev.reset_index()[["event_id", "entity_id", "timestamp"]].copy()
        order["flagged"] = flagged.to_numpy()
        order = order.sort_values(["entity_id", "timestamp"])
        k = int(self.cfg["persistence_window_events"])
        order["persistence"] = order.groupby("entity_id")["flagged"].transform(
            lambda s: s.astype(float).rolling(k, min_periods=1).mean()
        )
        order["recurrence"] = (
            order.groupby("entity_id")["flagged"].transform("sum") / self.cfg["recurrence_cap"]
        ).clip(upper=1.0)
        persistence = order.set_index("event_id")["persistence"].reindex(ev.index)
        recurrence = order.set_index("event_id")["recurrence"].reindex(ev.index)

        w = self.cfg["weights"]
        wsum = w["detection"] + w["severity_hint"] + w["persistence"] + w["recurrence"]
        raw = (
            w["detection"] * detection
            + w["severity_hint"] * hint
            + w["persistence"] * persistence
            + w["recurrence"] * recurrence
        ) / wsum

        score = (100.0 * raw * crit).clip(lower=0.0, upper=100.0).round(1)
        score.name = "risk_score"
        return score
