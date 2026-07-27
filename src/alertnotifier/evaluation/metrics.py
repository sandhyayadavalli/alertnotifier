"""Evaluation harness: score detector output against the ground-truth labels.

Because the synthetic data is labelled, we can report real precision / recall /
false-positive rate per detector — the whole reason the labels exist. Two views:
per-detector (each rule vs the type it targets) and system-level (any rule fired
vs any anomaly).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Metrics:
    name: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn


def score(name: str, flagged: pd.Series, positive: pd.Series) -> Metrics:
    """``flagged`` and ``positive`` are boolean Series indexed by event_id."""
    flagged, positive = flagged.align(positive, join="inner")
    flagged = flagged.fillna(False).astype(bool)
    positive = positive.fillna(False).astype(bool)
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    fn = int((~flagged & positive).sum())
    tn = int((~flagged & ~positive).sum())
    return Metrics(name, tp, fp, fn, tn)


def format_report(rows: list[Metrics]) -> str:
    header = (
        f"{'detector':32s} {'precision':>9s} {'recall':>7s} {'fp_rate':>8s} "
        f"{'f1':>5s} {'support':>8s}"
    )
    lines = [header, "-" * len(header)]
    for m in rows:
        lines.append(
            f"{m.name:32s} {m.precision:9.1%} {m.recall:7.1%} {m.fpr:8.2%} "
            f"{m.f1:5.2f} {m.support:8d}"
        )
    return "\n".join(lines)


def pr_curve(scores: pd.Series, positive: pd.Series):
    """Precision-recall curve for a continuous score. Returns (precision,
    recall, thresholds, average_precision)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    s, p = scores.align(positive, join="inner")
    y = p.fillna(False).astype(int).to_numpy()
    x = s.fillna(s.min()).to_numpy()
    precision, recall, thr = precision_recall_curve(y, x)
    return precision, recall, thr, average_precision_score(y, x)
