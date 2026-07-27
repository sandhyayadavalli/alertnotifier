"""Shared event schema for AlertNotifier.

Every source emits the same ``Event`` shape, so the engine's detection /
scoring / correlation / escalation layers stay domain-agnostic and operate ONLY
on these fields.

The ground-truth anomaly labels deliberately live *outside* the ``Event`` (as
the ``gt_*`` dataset columns produced by the generators) so that detectors never
see the answer key — exactly as in production, where you don't get told which
events are anomalous.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SeverityHint(str, Enum):
    """Coarse, *noisy* hint emitted by the source system (e.g. an API-gateway
    risk flag). Useful as a scoring input, but NOT the ground truth — it is
    intentionally only loosely correlated with it."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Event(BaseModel):
    """A single observation from any source.

    ``attributes`` holds the source-specific payload. Keeping it a free-form
    dict is what lets one schema serve any domain.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    timestamp: datetime
    entity_id: str
    entity_type: str
    source: str
    event_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    severity_hint: SeverityHint = SeverityHint.LOW

    @classmethod
    def new(cls, **kwargs: Any) -> "Event":
        """Construct an event, auto-generating ``event_id`` if not supplied."""
        kwargs.setdefault("event_id", str(uuid.uuid4()))
        return cls(**kwargs)
