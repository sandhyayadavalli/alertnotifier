#!/usr/bin/env python
"""Run the rule detectors over the dataset and print the evaluation report.

Run:  python scripts/run_detectors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.dataset import load_dataset, split_events_labels  # noqa: E402
from alertnotifier.detectors.rules import _robust_z, default_detectors  # noqa: E402
from alertnotifier.evaluation.metrics import format_report, pr_curve, score  # noqa: E402


def main() -> None:
    df = load_dataset("api_events")
    events, labels = split_events_labels(df)
    labels = labels.set_index("event_id")
    gt_type = labels["gt_anomaly_type"]
    is_anom = labels["gt_is_anomaly"].astype(bool)

    detectors = default_detectors()
    fired = {
        d.name: d.detect(events).reindex(is_anom.index).fillna(False).astype(bool)
        for d in detectors
    }

    # --- per-detector: each rule vs the type it targets ---
    rows = [score(d.name, fired[d.name], gt_type == d.target) for d in detectors]
    print("Per-detector (each rule vs the anomaly type it targets):")
    print(format_report(rows))

    # --- system level: any rule fired vs any anomaly ---
    any_fired = pd.concat(fired.values(), axis=1).any(axis=1)
    print("\nSystem-level (any rule fires vs any anomaly):")
    print(format_report([score("OVERALL (any rule)", any_fired, is_anom)]))

    # --- recall by anomaly type -> exposes the multivariate gap the ML must close ---
    print("\nRecall by anomaly type (caught by at least one rule):")
    for t in sorted(gt_type.dropna().unique()):
        ids = gt_type[gt_type == t].index
        caught = int(any_fired.reindex(ids).fillna(False).sum())
        flag = "   <-- held out for ML" if t == "multivariate_outlier" else ""
        print(f"  {t:22s} {caught / len(ids):6.1%}  ({caught}/{len(ids)}){flag}")

    # --- why per-client baselines matter: naive global night rule vs our per-client rule ---
    ev_idx = events.set_index("event_id")
    naive_night = ev_idx["hour"].isin([0, 1, 2, 3, 4, 5])
    off_target = gt_type == "off_baseline"
    print("\nWhy per-client baselines matter (off_baseline):")
    print(format_report([
        score("off_baseline (naive night rule)", naive_night, off_target),
        score("off_baseline (per-client rule)", fired["off_baseline"], off_target),
    ]))

    # --- precision-recall curve for value_spike's continuous score ---
    g = ev_idx.groupby("entity_id")
    vs_score = pd.concat(
        [g["latency_ms"].transform(_robust_z), g["payload_bytes"].transform(_robust_z)], axis=1
    ).max(axis=1)
    precision, recall, _, ap = pr_curve(vs_score, gt_type == "value_spike")
    _save_pr_curve(precision, recall, ap)


def _save_pr_curve(precision, recall, ap: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(__file__).resolve().parents[1] / "docs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, color="#2563eb")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f"value_spike PR curve (AP = {ap:.2f})")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out / "value_spike_pr_curve.png"
    plt.savefig(path, dpi=120)
    print(f"\nPR curve (AP={ap:.2f}) -> {path.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
