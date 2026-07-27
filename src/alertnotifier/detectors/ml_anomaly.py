"""IsolationForest anomaly detector — the statistical layer.

Unsupervised: it fits on engineered numeric features with **no labels** and
scores each event by how easily it can be isolated (short average tree path =
anomaly). Its job is the anomaly the rules can't express — the scraper
(`multivariate_outlier`) — which it discovers from the *joint* feature space
rather than a hand-coded threshold.

Feature choices are deliberate: the raw signals that make the scraper jointly
rare (endpoint diversity + a fresh IP + small payloads, all at normal rate) are
present, while the pre-computed flags that the rules already handle perfectly
(`off_baseline_hour`, tier change) are left OUT — so the model earns the scraper
on its own and the rules/ML split stays complementary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

#: engineered features, in order (skewed magnitudes are log-compressed)
FEATURES = [
    "log_latency_ms",
    "log_payload_bytes",
    "log_req_count_5m",
    "distinct_endpoints_5m",
    "log_minutes_since_last_request",
    "is_error",
    "new_ip",
]


class IsolationForestDetector:
    name = "isolation_forest"

    def __init__(self, contamination: float = 0.10, n_estimators: int = 200, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def features(self, events: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=events["event_id"].to_numpy())
        f["log_latency_ms"] = np.log1p(events["latency_ms"].to_numpy())
        f["log_payload_bytes"] = np.log1p(events["payload_bytes"].to_numpy())
        f["log_req_count_5m"] = np.log1p(events["req_count_5m"].to_numpy())
        f["distinct_endpoints_5m"] = events["distinct_endpoints_5m"].to_numpy()
        f["log_minutes_since_last_request"] = np.log1p(events["minutes_since_last_request"].to_numpy())
        f["is_error"] = events["is_error"].astype(int).to_numpy()
        f["new_ip"] = events["new_ip"].astype(int).to_numpy()
        return f[FEATURES]

    def fit(self, events: pd.DataFrame) -> "IsolationForestDetector":
        self.model.fit(self.features(events).to_numpy())
        return self

    def score(self, events: pd.DataFrame) -> pd.Series:
        """Anomaly score per event — higher = more anomalous."""
        X = self.features(events)
        return pd.Series(-self.model.score_samples(X.to_numpy()), index=X.index, name="if_score")

    def detect(self, events: pd.DataFrame) -> pd.Series:
        """Boolean flag at the contamination operating point."""
        X = self.features(events)
        return pd.Series(self.model.predict(X.to_numpy()) == -1, index=X.index, name=self.name)
