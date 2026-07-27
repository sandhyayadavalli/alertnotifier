"""Dataset I/O helpers shared by the generators and the evaluation harness.

The on-disk dataset is a flat table: the core ``Event`` fields, the flattened
``attributes``, and two ground-truth columns (``gt_is_anomaly``,
``gt_anomaly_type``). ``split_events_labels`` enforces the separation so the
detection code can only ever load the event view, never the labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from alertnotifier.models import Event

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "generated"

GT_COLS = ["gt_is_anomaly", "gt_anomaly_type"]

# (event, is_anomaly, anomaly_type)
LabeledEvent = tuple[Event, bool, Optional[str]]


def labeled_events_to_dataframe(records: Iterable[LabeledEvent]) -> pd.DataFrame:
    """Flatten labelled events into a tabular DataFrame, ordered chronologically
    so it reads like a real event stream (what the replay harness will consume)."""
    rows = []
    for ev, is_anom, anom_type in records:
        rows.append(
            {
                "event_id": ev.event_id,
                "timestamp": ev.timestamp,
                "entity_id": ev.entity_id,
                "entity_type": ev.entity_type,
                "source": ev.source,
                "event_type": ev.event_type,
                "severity_hint": ev.severity_hint.value,
                **ev.attributes,  # flattened domain-specific payload
                "gt_is_anomaly": is_anom,
                "gt_anomaly_type": anom_type,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").reset_index(drop=True)


def write_dataset(df: pd.DataFrame, name: str, out_dir: Path = DATA_DIR) -> tuple[Path, str]:
    """Write ``df`` as both CSV and Parquet. Returns (csv_path, parquet_status)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    parquet_status = "written"
    try:
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    except Exception as exc:  # pyarrow not installed, etc. — CSV still works
        parquet_status = f"skipped ({exc.__class__.__name__})"
    return csv_path, parquet_status


def load_dataset(name: str, out_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a generated dataset (prefers Parquet, falls back to CSV)."""
    pq = out_dir / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(out_dir / f"{name}.csv", parse_dates=["timestamp"])


def split_events_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (events_view, labels_view).

    ``events_view`` has the ground-truth columns removed — this is the ONLY
    view detectors should ever touch. ``labels_view`` keys labels by event_id
    for the evaluation harness.
    """
    events = df.drop(columns=[c for c in GT_COLS if c in df.columns])
    labels = df[["event_id", *[c for c in GT_COLS if c in df.columns]]].copy()
    return events, labels
