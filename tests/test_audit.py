"""The audit log is append-only and hash-chained: an intact chain verifies, and
any edit or deletion of a past row is detected."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sqlalchemy import text  # noqa: E402

from alertnotifier.audit.log import GENESIS, AuditLog  # noqa: E402


def _log(n: int = 5) -> AuditLog:
    log = AuditLog(":memory:")
    log.extend([{
        "incident_id": f"INC-{i:03d}", "entity_id": "CLI-0001", "start_ts": "t0", "end_ts": "t1",
        "alert_count": 2, "detectors": "rate_anomaly", "risk_score": 50.0 + i,
        "action": "notify_analyst", "decided_at": f"2026-06-01T00:0{i}:00", "actor": "engine",
    } for i in range(n)])
    return log


def test_intact_chain_verifies():
    ok, bad = _log().verify()
    assert ok and bad is None


def test_each_row_links_to_the_previous():
    rows = _log().entries()
    assert rows[0].prev_hash == GENESIS
    for i in range(1, len(rows)):
        assert rows[i].prev_hash == rows[i - 1].row_hash


def test_editing_a_row_is_detected():
    log = _log()
    with log.engine.begin() as c:
        c.execute(text("UPDATE audit_log SET risk_score = 999 WHERE seq = 3"))
    ok, bad = log.verify()
    assert not ok and bad == 3


def test_deleting_a_row_is_detected():
    log = _log()
    with log.engine.begin() as c:
        c.execute(text("DELETE FROM audit_log WHERE seq = 3"))
    ok, bad = log.verify()
    assert not ok  # row 4's prev_hash no longer matches the (now-missing) row 3
