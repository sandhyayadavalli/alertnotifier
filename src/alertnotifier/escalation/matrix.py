"""Escalation matrix — route an incident to an action by its risk score.

Bands and actions come from ``config/escalation_matrix.yaml`` so the policy is
tunable without code. Auto-actions are stubbed: ``apply`` just records the
decision on the incident and flips its lifecycle status.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from alertnotifier.correlation.incidents import Incident

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "escalation_matrix.yaml"


class EscalationMatrix:
    def __init__(self, config: dict):
        self.bands = sorted(config["bands"], key=lambda b: b["min"])
        self.actions = config.get("actions", {})

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "EscalationMatrix":
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def route(self, score: float) -> dict:
        """Return the matching band dict for a score."""
        for band in self.bands:
            if band["min"] <= score < band["max"]:
                return band
        return self.bands[-1]  # scores at/over the top fall in the last band

    def apply(self, incident: Incident) -> Incident:
        """Assign the action + SLA to an incident and update its lifecycle."""
        band = self.route(incident.score)
        incident.action = band["action"]
        incident.sla_minutes = band.get("sla_minutes")
        incident.status = "resolved" if band["action"] == "log_only" else "escalated"
        return incident

    def action_description(self, action: str) -> str:
        return self.actions.get(action, action)
