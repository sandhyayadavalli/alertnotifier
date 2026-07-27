#!/usr/bin/env python
"""Fit the IsolationForest and measure it — especially on the held-out scraper
(`multivariate_outlier`) that the rules catch 0% of.

Run:  python scripts/run_ml.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.dataset import load_dataset, split_events_labels  # noqa: E402
from alertnotifier.detectors.ml_anomaly import IsolationForestDetector  # noqa: E402
from alertnotifier.detectors.rules import default_detectors  # noqa: E402
from alertnotifier.evaluation.metrics import format_report, pr_curve, score  # noqa: E402


def _recall_by_type(flag: pd.Series, gt_type: pd.Series) -> None:
    for t in sorted(gt_type.dropna().unique()):
        ids = gt_type[gt_type == t].index
        hit = flag.reindex(ids).fillna(False)
        star = "   <-- was 0% with rules" if t == "multivariate_outlier" else ""
        print(f"  {t:22s} {hit.mean():6.1%}  ({int(hit.sum())}/{len(ids)}){star}")


def main() -> None:
    df = load_dataset("api_events")
    events, labels = split_events_labels(df)
    labels = labels.set_index("event_id")
    gt_type = labels["gt_anomaly_type"]
    is_anom = labels["gt_is_anomaly"].astype(bool)

    # --- fit IsolationForest (unsupervised — no labels touched) ---
    iso = IsolationForestDetector(contamination=0.10).fit(events)
    if_flag = iso.detect(events).reindex(is_anom.index).fillna(False)
    if_score = iso.score(events).reindex(is_anom.index)

    # --- rule system, for the before/after ---
    rule_fired = pd.concat(
        [d.detect(events).reindex(is_anom.index).fillna(False) for d in default_detectors()], axis=1
    ).any(axis=1)
    combined = rule_fired | if_flag

    print("IsolationForest as a general detector (vs any anomaly):")
    print(format_report([score("isolation_forest", if_flag, is_anom)]))

    # --- HEADLINE: the held-out scraper, rules vs ML ---
    scraper = gt_type == "multivariate_outlier"
    precision, recall, _, ap = pr_curve(if_score, scraper)
    print("\nHeld-out scraper (multivariate_outlier) — rules vs ML:")
    print(format_report([
        score("scraper: rules", rule_fired, scraper),
        score("scraper: IsolationForest", if_flag, scraper),
    ]))
    print(f"  IsolationForest scraper-ranking AP = {ap:.2f}  (thresholds aside, how well it ranks)")

    print("\nIsolationForest recovery by anomaly type (recall at operating point):")
    _recall_by_type(if_flag, gt_type)
    print("  (weak on drift/off_baseline/status_change by design — those are the rules' job)")

    print("\nSystem level — rules only vs rules + IsolationForest (vs any anomaly):")
    print(format_report([
        score("rules only", rule_fired, is_anom),
        score("rules + IsolationForest", combined, is_anom),
    ]))

    print("\nRecall by anomaly type — rules + IsolationForest (union):")
    _recall_by_type(combined, gt_type)

    _save_pr(precision, recall, ap)


def _save_pr(precision, recall, ap: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(__file__).resolve().parents[1] / "docs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, color="#dc2626")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f"scraper detection by IsolationForest (AP = {ap:.2f})")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out / "scraper_if_pr_curve.png"
    plt.savefig(path, dpi=120)
    print(f"\nPR curve -> {path.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
