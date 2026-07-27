"""The IsolationForest layer should isolate the multivariate (scraper) pattern
that no single-feature rule targets — high endpoint diversity + fresh IP + small
payloads, all at a normal request rate."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.detectors.ml_anomaly import IsolationForestDetector  # noqa: E402


def _mixed_traffic() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(150):  # normal clients: one endpoint, stable IP, larger payloads
        rows.append(dict(
            event_id=f"n{i}", latency_ms=float(rng.normal(100, 20)),
            payload_bytes=float(rng.normal(2000, 300)), req_count_5m=1,
            distinct_endpoints_5m=1, minutes_since_last_request=float(rng.uniform(1, 30)),
            is_error=False, new_ip=False,
        ))
    for j in range(20):  # scraper: many endpoints, fresh IP, small payloads, normal rate
        rows.append(dict(
            event_id=f"s{j}", latency_ms=float(rng.normal(95, 15)),
            payload_bytes=float(rng.normal(300, 50)), req_count_5m=4,
            distinct_endpoints_5m=7, minutes_since_last_request=float(rng.uniform(1, 3)),
            is_error=False, new_ip=True,
        ))
    return pd.DataFrame(rows)


def test_isolation_forest_flags_scraper_pattern():
    df = _mixed_traffic()
    flag = IsolationForestDetector(contamination=0.15, random_state=0).fit(df).detect(df)
    scraper = flag[[i.startswith("s") for i in flag.index]]
    normal = flag[[i.startswith("n") for i in flag.index]]
    assert scraper.mean() >= 0.6   # most scrapers caught
    assert normal.mean() <= 0.2    # few normals falsely flagged


def test_scores_rank_scrapers_above_normal():
    df = _mixed_traffic()
    s = IsolationForestDetector(random_state=0).fit(df).score(df)
    scraper = s[[i.startswith("s") for i in s.index]].mean()
    normal = s[[i.startswith("n") for i in s.index]].mean()
    assert scraper > normal        # higher anomaly score for scrapers
