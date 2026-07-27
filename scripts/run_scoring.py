#!/usr/bin/env python
"""Score every event 0-100 with the config-driven risk engine, and show that
real anomalies score high while normal traffic stays low.

Run:  python scripts/run_scoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.dataset import load_dataset, split_events_labels  # noqa: E402
from alertnotifier.detectors.ml_anomaly import IsolationForestDetector  # noqa: E402
from alertnotifier.detectors.rules import default_detectors  # noqa: E402
from alertnotifier.evaluation.metrics import pr_curve  # noqa: E402
from alertnotifier.scoring.engine import RiskScorer  # noqa: E402


def main() -> None:
    df = load_dataset("api_events")
    events, labels = split_events_labels(df)
    labels = labels.set_index("event_id")
    gt_type = labels["gt_anomaly_type"]
    is_anom = labels["gt_is_anomaly"].astype(bool)

    # assemble the per-event signals: rule flags + ML flag
    fired = {d.name: d.detect(events).reindex(events["event_id"]).fillna(False) for d in default_detectors()}
    iso = IsolationForestDetector(contamination=0.10).fit(events)
    fired["isolation_forest"] = iso.detect(events).reindex(events["event_id"]).fillna(False)
    fired = pd.DataFrame(fired)

    risk = RiskScorer.from_yaml().score(events, fired).reindex(is_anom.index)

    print("Risk score by group (0-100):")
    print(f"  {'normal':24s} median {risk[~is_anom].median():5.1f}   mean {risk[~is_anom].mean():5.1f}")
    for t in sorted(gt_type.dropna().unique()):
        r = risk[gt_type == t]
        print(f"  {t:24s} median {r.median():5.1f}   mean {r.mean():5.1f}")

    _, _, _, ap = pr_curve(risk, is_anom)
    print(f"\nRisk score as a single ranker: average precision = {ap:.2f}")

    bands = [(0, 25, "info"), (25, 50, "low"), (50, 75, "medium"), (75, 101, "high")]
    print("\nEvents per risk band (this is what Phase 5 will escalate on):")
    print(f"  {'band':18s} {'normal':>9s} {'anomaly':>9s}")
    for lo, hi, name in bands:
        m = (risk >= lo) & (risk < hi)
        label = f"{name} [{lo}-{hi - 1 if hi <= 100 else 100}]"
        print(f"  {label:18s} {int((m & ~is_anom).sum()):9d} {int((m & is_anom).sum()):9d}")

    _save_hist(risk, is_anom)


def _save_hist(risk: pd.Series, is_anom: pd.Series) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(__file__).resolve().parents[1] / "docs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(risk[~is_anom], bins=40, alpha=0.6, label="normal", color="#2563eb", density=True)
    plt.hist(risk[is_anom], bins=40, alpha=0.6, label="anomaly", color="#dc2626", density=True)
    plt.xlabel("risk score (0-100)")
    plt.ylabel("density")
    plt.legend()
    plt.title("Risk score: normal vs anomaly")
    plt.tight_layout()
    path = out / "risk_score_distribution.png"
    plt.savefig(path, dpi=120)
    print(f"\nhistogram -> {path.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
