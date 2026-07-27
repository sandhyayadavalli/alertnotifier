#!/usr/bin/env python
"""Generate a synthetic API / client-abuse dataset with ground-truth labels.

Theme: each entity is an **API client** (an API key / app). Each event is one
API request. The engine watches these requests for the six generic anomaly
types (PLAN.md Section 4), here instantiated as:

    value_spike          -> a single request with abnormally high latency/payload
    sustained_drift      -> a client's error ratio stays high across a window
                            (isolated errors are normal noise -> rule FP bait)
    rate_anomaly         -> a request-rate burst (scraping / DoS) OR a client
                            going silent for a long gap
    off_baseline         -> activity at an hour normal globally but abnormal for
                            THIS client; 24/7 clients are excluded, so a naive
                            global night-hours rule over-fires -> jointly rare
    status_change        -> the client's access tier flips unexpectedly
    multivariate_outlier -> a scraper signature: many distinct endpoints + a
                            fresh IP + small payloads. Each feature is normal
                            alone but jointly rare -> the ML layer's job, held
                            out from the rules (Phase 3)

Run:  python scripts/generate_events.py
Out:  data/generated/api_events.{csv,parquet}
"""
from __future__ import annotations

import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.dataset import labeled_events_to_dataframe, write_dataset  # noqa: E402
from alertnotifier.models import Event, SeverityHint  # noqa: E402

SEED = 42
N_CLIENTS = 60
DAYS = 7
START = datetime(2026, 6, 1, tzinfo=timezone.utc)
ENDPOINTS = [
    "/v1/search", "/v1/users", "/v1/orders", "/v1/products",
    "/v1/auth", "/v1/reports", "/v1/files", "/v1/admin",
]
TIERS = ["throttled", "standard", "trusted"]  # low -> high access


def make_profile(idx: int, rng: np.random.Generator, faker: Faker) -> dict:
    always_on = bool(rng.random() < 0.25)
    if always_on:
        active_hours = set(range(24))
    else:
        start = int(rng.integers(6, 10))
        active_hours = {h % 24 for h in range(start, start + int(rng.integers(8, 12)))}
    return {
        "uid": f"CLI-{idx:04d}",
        "home_ip": faker.ipv4_public(),
        "base_tier": TIERS[int(rng.choice([0, 1, 1, 1, 2]))],   # mostly standard
        "endpoints": [str(e) for e in rng.choice(ENDPOINTS, size=int(rng.integers(2, 5)), replace=False)],
        "base_error_rate": float(rng.uniform(0.01, 0.05)),
        "latency_mu": float(rng.uniform(40, 160)),              # ms
        "payload_mu": float(rng.uniform(400, 4000)),            # bytes
        "base_rate": float(rng.uniform(1, 7)),                  # requests / active hour
        "always_on": always_on,
        "active_hours": active_hours,
    }


def make_request(p: dict, ts: datetime, rng: np.random.Generator) -> dict:
    err = rng.random() < p["base_error_rate"]
    return {
        "ts": ts,
        "endpoint": str(rng.choice(p["endpoints"])),
        "method": "POST" if rng.random() < 0.3 else "GET",
        "status_code": int(rng.choice([500, 429, 403])) if err else 200,
        "latency_ms": round(float(max(1.0, rng.normal(p["latency_mu"], p["latency_mu"] * 0.3))), 1),
        "payload_bytes": int(max(50, rng.normal(p["payload_mu"], p["payload_mu"] * 0.4))),
        "client_ip": p["home_ip"],
        "tier": p["base_tier"],
        "dropped": False,
        "gt_type": None,
    }


def generate_client(idx: int, rng: np.random.Generator, faker: Faker) -> list:
    p = make_profile(idx, rng, faker)

    # --- baseline traffic: Poisson arrivals during active hours ---
    reqs = []
    for day in range(DAYS):
        for hour in sorted(p["active_hours"]):
            for _ in range(int(rng.poisson(p["base_rate"]))):
                ts = START + timedelta(
                    days=day, hours=hour,
                    minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)),
                )
                reqs.append(make_request(p, ts, rng))
    reqs.sort(key=lambda r: r["ts"])
    n = len(reqs)
    if n < 20:                       # too quiet to inject into meaningfully
        return _build(p, reqs, rng)

    # --- value_spike: labelled extreme latency/payload ---
    for i in rng.choice(n, size=max(1, int(0.015 * n)), replace=False):
        r = reqs[int(i)]
        if r["gt_type"]:
            continue
        if rng.random() < 0.5:
            r["latency_ms"] = round(r["latency_ms"] * float(rng.uniform(8, 20)), 1)
        else:
            r["payload_bytes"] = int(r["payload_bytes"] * float(rng.uniform(8, 20)))
        r["gt_type"] = "value_spike"
    # unlabelled "elevated but legit" latency -> false-positive bait for a naive threshold
    for i in rng.choice(n, size=max(1, int(0.01 * n)), replace=False):
        r = reqs[int(i)]
        if not r["gt_type"]:
            r["latency_ms"] = round(r["latency_ms"] * float(rng.uniform(3, 5)), 1)

    # --- sustained_drift: a contiguous window forced to errors ---
    if rng.random() < 0.5:
        s = int(rng.integers(0, n - 1))
        t0, span = reqs[s]["ts"], timedelta(hours=float(rng.uniform(2, 4)))
        for r in reqs[s:]:
            if r["ts"] - t0 > span:
                break
            r["status_code"] = int(rng.choice([500, 503, 429]))
            r["gt_type"] = r["gt_type"] or "sustained_drift"

    # --- silence: only for always-on clients (a gap is clearly off-baseline) ---
    if p["always_on"] and rng.random() < 0.6:
        s = int(rng.integers(0, n - 2))
        t0, gap = reqs[s]["ts"], timedelta(hours=float(rng.uniform(6, 24)))
        e = s
        while e < n and reqs[e]["ts"] - t0 <= gap:
            reqs[e]["dropped"] = True
            e += 1
        if e < n:
            reqs[e]["gt_type"] = reqs[e]["gt_type"] or "rate_anomaly"

    # --- status_change: tier flips for a contiguous run ---
    if rng.random() < 0.4:
        new_tier = str(rng.choice([t for t in TIERS if t != p["base_tier"]]))
        s = int(rng.integers(0, n - 1))
        t0, span = reqs[s]["ts"], timedelta(hours=float(rng.uniform(1, 3)))
        for r in reqs[s:]:
            if r["ts"] - t0 > span:
                break
            if r["dropped"]:
                continue
            r["tier"] = new_tier
            r["gt_type"] = r["gt_type"] or "status_change"

    # --- rate burst: many extra requests in a short window ---
    if rng.random() < 0.5:
        t0 = START + timedelta(
            days=int(rng.integers(0, DAYS)),
            hours=int(rng.choice(sorted(p["active_hours"]))),
            minutes=int(rng.integers(0, 40)),
        )
        for _ in range(int(rng.integers(20, 55))):
            r = make_request(p, t0 + timedelta(seconds=int(rng.integers(0, 1200))), rng)
            r["gt_type"] = "rate_anomaly"
            reqs.append(r)

    # --- off_baseline: off-hours activity (business clients only) ---
    if not p["always_on"] and rng.random() < 0.5:
        day = int(rng.integers(0, DAYS))
        for _ in range(int(rng.integers(3, 10))):
            ts = START + timedelta(days=day, hours=int(rng.choice([1, 2, 3, 4])),
                                   minutes=int(rng.integers(0, 60)))
            r = make_request(p, ts, rng)
            r["gt_type"] = "off_baseline"
            reqs.append(r)

    # --- multivariate_outlier: scraper signature (many endpoints, fresh IP, small payloads) ---
    if rng.random() < 0.5:
        scraper_ip = faker.ipv4_public()
        t0 = START + timedelta(
            days=int(rng.integers(0, DAYS)),
            hours=int(rng.choice(sorted(p["active_hours"]))),
            minutes=int(rng.integers(0, 30)),
        )
        t = t0
        for _ in range(int(rng.integers(15, 35))):
            r = make_request(p, t, rng)
            r["endpoint"] = str(rng.choice(ENDPOINTS))            # enumerates ALL endpoints
            r["payload_bytes"] = int(max(50, rng.normal(300, 80)))  # uniformly small
            r["status_code"] = 200
            r["client_ip"] = scraper_ip                           # fresh IP
            r["gt_type"] = "multivariate_outlier"
            reqs.append(r)
            t += timedelta(seconds=int(rng.integers(45, 90)))     # slow -> rate stays normal

    return _build(p, reqs, rng)


def _build(p: dict, reqs: list, rng: np.random.Generator) -> list:
    """Drop silenced requests, then compute rolling per-client features and
    emit validated Event objects."""
    reqs = [r for r in reqs if not r["dropped"]]
    reqs.sort(key=lambda r: r["ts"])
    window: deque = deque()
    records, prev = [], None
    for r in reqs:
        while window and (r["ts"] - window[0]["ts"]) > timedelta(minutes=5):
            window.popleft()
        window.append(r)
        gap_min = (r["ts"] - prev["ts"]).total_seconds() / 60.0 if prev else 0.0
        is_error = r["status_code"] >= 400

        if r["status_code"] >= 500 or r["latency_ms"] > 1000:
            sev = SeverityHint.HIGH if rng.random() < 0.5 else SeverityHint.MEDIUM
        elif is_error:
            sev = SeverityHint.MEDIUM if rng.random() < 0.5 else SeverityHint.LOW
        else:
            sev = SeverityHint.LOW

        ev = Event.new(
            timestamp=r["ts"], entity_id=p["uid"], entity_type="api_client",
            source="api", event_type="api_request", severity_hint=sev,
            attributes={
                "endpoint": r["endpoint"],
                "method": r["method"],
                "status_code": r["status_code"],
                "is_error": bool(is_error),
                "latency_ms": r["latency_ms"],
                "payload_bytes": r["payload_bytes"],
                "client_ip": r["client_ip"],
                "new_ip": bool(r["client_ip"] != p["home_ip"]),
                "minutes_since_last_request": round(gap_min, 1),
                "req_count_5m": len(window),
                "distinct_endpoints_5m": len({w["endpoint"] for w in window}),
                "client_tier": r["tier"],
                "base_client_tier": p["base_tier"],
                "always_on": p["always_on"],
                "hour": r["ts"].hour,
                "off_baseline_hour": bool(r["ts"].hour not in p["active_hours"]),
            },
        )
        records.append((ev, r["gt_type"] is not None, r["gt_type"]))
        prev = r
    return records


def _summary(df, csv_path, parquet_status) -> None:
    n = len(df)
    n_anom = int(df["gt_is_anomaly"].sum())
    always_on = df.groupby("entity_id")["always_on"].first().sum()
    print(f"api: {n} requests from {df['entity_id'].nunique()} clients "
          f"({int(always_on)} always-on, {df['entity_id'].nunique() - int(always_on)} business-hours)")
    print(f"  anomalies: {n_anom} ({n_anom / n:.1%})")
    for kind, count in df.loc[df["gt_is_anomaly"], "gt_anomaly_type"].value_counts().items():
        print(f"    {kind:22s} {count:4d}  ({count / n:.1%})")
    print(f"  -> {csv_path}  (parquet: {parquet_status})")


def main() -> None:
    faker = Faker()
    Faker.seed(SEED)
    records = []
    for i in range(1, N_CLIENTS + 1):
        records += generate_client(i, np.random.default_rng(SEED + i), faker)
    df = labeled_events_to_dataframe(records)
    csv_path, parquet_status = write_dataset(df, "api_events")
    _summary(df, csv_path, parquet_status)


if __name__ == "__main__":
    main()
