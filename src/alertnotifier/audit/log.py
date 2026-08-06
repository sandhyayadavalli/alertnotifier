"""Append-only, hash-chained audit log — a tamper-evident record of decisions.

Every row stores a SHA-256 hash of (the previous row's hash + this row's
canonical content). Editing any past row changes its hash, so it no longer
matches what the *next* row recorded as its ``prev_hash`` — and ``verify()`` can
pinpoint the first tampered (or deleted) row. Backed by SQLite via SQLAlchemy.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import Column, Float, Integer, String, create_engine, select
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy.pool import StaticPool

Base = declarative_base()
GENESIS = "0" * 64


class AuditRow(Base):
    __tablename__ = "audit_log"
    seq = Column(Integer, primary_key=True)
    incident_id = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    start_ts = Column(String, nullable=False)
    end_ts = Column(String, nullable=False)
    alert_count = Column(Integer, nullable=False)
    detectors = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False)
    action = Column(String, nullable=False)
    decided_at = Column(String, nullable=False)
    actor = Column(String, nullable=False)
    prev_hash = Column(String, nullable=False)
    row_hash = Column(String, nullable=False)


def _payload(seq, incident_id, entity_id, start_ts, end_ts, alert_count,
             detectors, risk_score, action, decided_at, actor) -> str:
    """Canonical, order-stable JSON of a row's content (excludes the hashes)."""
    return json.dumps(
        {"seq": seq, "incident_id": incident_id, "entity_id": entity_id, "start_ts": start_ts,
         "end_ts": end_ts, "alert_count": alert_count, "detectors": detectors,
         "risk_score": risk_score, "action": action, "decided_at": decided_at, "actor": actor},
        sort_keys=True, separators=(",", ":"),
    )


def _chain_hash(prev_hash: str, payload: str) -> str:
    return hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, db_path: str = ":memory:"):
        if db_path == ":memory:":
            # StaticPool keeps one connection so the in-memory DB persists across sessions
            self.engine = create_engine(
                "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
            )
        else:
            self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self._prev_hash, self._seq = self._tail()

    def _tail(self) -> tuple[str, int]:
        with Session(self.engine) as s:
            last = s.execute(select(AuditRow).order_by(AuditRow.seq.desc())).scalars().first()
            return (last.row_hash, last.seq) if last else (GENESIS, 0)

    def append(self, *, decided_at: Optional[str] = None, actor: str = "engine", **fields) -> str:
        """Append a single decision; returns its row hash."""
        self.extend([{**fields, "decided_at": decided_at, "actor": actor}])
        return self._prev_hash

    def extend(self, records: Iterable[dict]) -> None:
        """Append many decisions in one transaction (each still hash-chained)."""
        with Session(self.engine) as s:
            for rec in records:
                rec = dict(rec)
                decided_at = rec.pop("decided_at", None) or datetime.now(timezone.utc).isoformat()
                actor = rec.pop("actor", None) or "engine"
                self._seq += 1
                payload = _payload(self._seq, rec["incident_id"], rec["entity_id"], rec["start_ts"],
                                   rec["end_ts"], rec["alert_count"], rec["detectors"],
                                   rec["risk_score"], rec["action"], decided_at, actor)
                row_hash = _chain_hash(self._prev_hash, payload)
                s.add(AuditRow(seq=self._seq, prev_hash=self._prev_hash, row_hash=row_hash,
                               decided_at=decided_at, actor=actor, **rec))
                self._prev_hash = row_hash
            s.commit()

    def verify(self) -> tuple[bool, Optional[int]]:
        """Recompute the whole chain. Returns (ok, first_bad_seq_or_None)."""
        with Session(self.engine) as s:
            prev = GENESIS
            for r in s.execute(select(AuditRow).order_by(AuditRow.seq)).scalars():
                payload = _payload(r.seq, r.incident_id, r.entity_id, r.start_ts, r.end_ts,
                                   r.alert_count, r.detectors, r.risk_score, r.action, r.decided_at, r.actor)
                if r.prev_hash != prev or r.row_hash != _chain_hash(prev, payload):
                    return False, r.seq
                prev = r.row_hash
        return True, None

    def entries(self, limit: Optional[int] = None) -> list[AuditRow]:
        with Session(self.engine) as s:
            q = select(AuditRow).order_by(AuditRow.seq)
            if limit:
                q = q.limit(limit)
            return list(s.execute(q).scalars())

    def count(self) -> int:
        with Session(self.engine) as s:
            return len(list(s.execute(select(AuditRow.seq)).scalars()))
