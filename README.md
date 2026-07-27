# AlertNotifier

A domain-agnostic **real-time risk monitoring & escalation engine**. One
pipeline — **ingest → detect → score → correlate → escalate → audit →
dashboard** — that watches a stream of entity events, flags the risky ones, and
records every decision. Point it at any source (a synthetic generator now; a
CSV, log stream, or API later) without touching the engine. See
[PLAN.md](PLAN.md) for the full build plan and the genric anomaly taxonomy.

> The default dataset is **synthetic**. What the project demonstrates is the
> engineering: a clean pipeline, detection you can *measure* (real
> precision/recall, since the data is labelled), incident correlation,
> tamper-evident auditing, and a live dashboard.

## Status

- [x] **Reusable core** — domain-agnostic `Event` schema + dataset I/O with
      ground-truth-label separation (detectors never see the labels)
- [x] Phase 1 — Generic synthetic data generator (API / client-abuse theme)
- [x] Phase 2 — Rule detectors + evaluation harness
- [x] Phase 3 — IsolationForest (99.7% scraper recovery vs rules' 0%)
- [x] Phase 4 — Risk scoring engine (0–100, config-driven)
- [ ] Phase 5 — Incident correlation + escalation matrix
- [ ] Phase 6 — Audit log + replay harness
- [ ] Phase 7 — Streamlit dashboard
- [ ] Phase 8 — Polish & packaging

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Generate data

```bash
python scripts/generate_events.py
```

Produces `data/generated/api_events.{csv,parquet}` — ~21k API-request events
from 60 clients (~12% labelled anomalies). Each row is an `Event`; the
`gt_is_anomaly` / `gt_anomaly_type` columns are **ground truth for evaluation
only** — detectors must never read them.

**Theme — API / client abuse.** Entities are API clients, each event is one
request. The six generic anomaly types map to: latency/payload spikes, sustained
error-rate drift, request-rate bursts or client silence, off-baseline access
hours, access-tier changes, and a scraper signature (many endpoints + fresh IP +
small payloads — the multivariate case for the ML layer).

## Evaluate the detectors

```bash
python scripts/run_detectors.py
```

Runs the rule detectors and prints precision / recall / false-positive rate per
detector against the labels, plus a PR curve (saved to `docs/eval/`). The rules
catch the clear-cut anomalies but 0% of the scraper case — the gap the ML layer
(Phase 3) closes.

```bash
python scripts/run_ml.py
```

Fits the IsolationForest and measures it against the held-out scraper the rules
miss entirely: it recovers **99.7%**, lifting the combined system's overall
recall from 55% to 90% (F1 0.67 → 0.82).

```bash
python scripts/run_scoring.py
```

Scores every event 0–100 (weights in `config/scoring_config.yaml`). Normal
traffic medians ~16, anomalies ~48–80 — a clean single dial for escalation.

## Layout

```
src/alertnotifier/   engine (models, sources, detectors, scoring, correlation, escalation, audit, evaluation)
scripts/            data generators
config/             scoring_config.yaml, escalation_matrix.yaml
dashboard/          Streamlit app
tests/              pytest suite
docs/               architecture + detector spec cards
```
