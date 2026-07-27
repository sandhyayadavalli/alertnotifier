# AlertNotifier — Real-Time Risk Monitoring & Escalation Engine

A domain-agnostic engine that watches a stream of **entity events**, flags the
risky ones, scores them, groups them into incidents, escalates by severity, and
records every decision in a tamper-evident audit trail — all visible on a live
dashboard. Point it at any source (a synthetic generator now; a CSV, log stream,
or API later) without changing the engine.

Built as a **portfolio project**: the value is a clean, well-tested pipeline and
detection you can actually *measure*, not a pile of features.

What makes it more than a toy:
1. **Measured, not vibes.** The synthetic data is labelled, so every detector
   reports precision / recall / false-positive rate. Thresholds are justified
   with numbers.
2. **Incident correlation + dedup.** Raw alerts are grouped into incidents with
   a lifecycle — the real operational problem (alert fatigue), not just firing.
3. **Honest "real-time."** A replay harness streams events through the engine at
   speed so the dashboard updates live.
4. **Defensible ML.** The IsolationForest is given anomaly types *held out from
   the rules* so it genuinely has something to find, then measured against the
   known labels.

---

## 1. Core Idea

One pipeline — **ingest → detect → score → correlate → escalate → audit →
dashboard** — that is completely agnostic to what it's monitoring. A
"monitored entity" is anything that emits events over time: a server, a user, a
device, an account, a sensor, a transaction stream. The engine only ever sees
the shared `Event` shape, so nothing downstream cares about the domain.

Instead of domain-specific detectors, the engine looks for a small set of
**generic anomaly archetypes** (Section 4) that show up in almost any
monitoring context.

---

## 2. Architecture

```
  Event source(s) ───────────►  Ingestion
  • synthetic generator (default)  • normalize to the Event schema
  • CSV / stream / API (later)     • replay harness (speed-controlled)
                                          │
                                          ▼
                              ┌────────────────────────────┐
                              │  Detection Layer            │
                              │   • rule detectors          │
                              │   • IsolationForest (ML)    │
                              └────────────────────────────┘
                                          │  measured vs. ground-truth labels
                                          ▼
                              ┌────────────────────────────┐
                              │  Risk Scoring (0–100)       │ ← scoring_config.yaml
                              └────────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────────┐
                              │  Incident Correlation       │  dedup + group by
                              │  + lifecycle                │  entity / time window
                              └────────────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────────┐
                              │  Escalation Matrix          │ ← escalation_matrix.yaml
                              │  log / notify / SLA / contain
                              └────────────────────────────┘
                                    │                  │
                                    ▼                  ▼
                           ┌───────────────┐   ┌────────────────────┐
                           │ Audit Log     │   │ Dashboard          │
                           │ (hash-chained)│   │ (Streamlit, live)  │
                           └───────────────┘   └────────────────────┘
```

---

## 3. Tech Stack

| Tech | Role |
|---|---|
| **Python 3.11+** (dev on 3.12) | Implementation language |
| **pandas / numpy** | Data shaping and feature engineering |
| **scikit-learn** | IsolationForest + metrics (precision/recall, PR curves) |
| **matplotlib / plotly** | Evaluation plots and dashboard charts |
| **Pydantic** | The shared `Event` schema |
| **SQLAlchemy + SQLite** | Event store + append-only audit log |
| **FastAPI + uvicorn** | `POST /events` live ingestion endpoint |
| **Streamlit** | Live dashboard |
| **PyYAML** | Config-driven scoring weights + escalation matrix |
| **Faker** | Synthetic entity/data generation |
| **pytest** | Detector / scoring / correlation / audit-chain tests |

---

## 4. Generic Anomaly Taxonomy

The heart of the project. Any entity stream tends to exhibit the same handful of
anomaly shapes — the engine detects these, not domain-specific ones:

| Type | What it is | Detected by |
|---|---|---|
| `value_spike` | a metric transiently exceeds its normal range | rule (threshold) |
| `sustained_drift` | a metric stays out of range across a window; one-off spikes are injected as **noise** (false-positive bait) | rule + window logic |
| `rate_anomaly` | an entity's event frequency spikes (burst) or drops to silence | rule (rate / gap) |
| `off_baseline` | a value normal *globally* but abnormal for **this entity's own** baseline; a naive global threshold over-fires | rule needing per-entity baselines |
| `status_change` | an unexpected categorical/state flip (status, tier, flag) | rule (state transition) |
| `multivariate_outlier` | individually-normal features that are **jointly** rare | **ML (IsolationForest)** — held out from the rules |

The last three are deliberately *jointly rare* so the statistical layer has real
work the rules can't easily do. (These map directly onto the anomalies already
prototyped in the earlier two-domain build, so that logic transfers.)

## Why ML, and Why IsolationForest

**Why ML at all.** Rules only catch patterns you can describe in advance
(`value > threshold`). But `multivariate_outlier` — a combination of features
that are each normal alone but rare together — can't be written as an
if-statement for a pattern you haven't seen. Unsupervised anomaly detection
catches the "I can't define it, but this is weird vs. normal" cases.

**Why IsolationForest.** Dictated by the problem shape: data is **unlabeled** in
the real world → unsupervised; you want **rare outliers**, not known buckets →
anomaly detection, not classification; features are **tabular/numeric** and it
must be **fast**. IsolationForest matches all of it — isolates outliers by random
partitioning (anomalies need fewer splits → shorter path length), needs no
labels, scales linearly, and has almost no knobs (mainly `contamination`), so
every setting is easy to justify.

| Alternative | Why not here |
|---|---|
| XGBoost / RF / neural net (supervised) | Need labels; only learn anomalies already labelled — can't catch the unknown |
| One-Class SVM | Also anomaly detection, but slower, scales poorly, sensitive to kernel/gamma |
| Local Outlier Factor | Density-based; doesn't apply cleanly to *new streaming* events |
| k-means / DBSCAN | Clustering isn't anomaly detection; assumes cluster structure |
| Autoencoders / deep learning | Overkill — needs big data, GPU, heavy tuning |
| z-score / IQR | That's what the *rules* already do — single-feature thresholds |

---

## 5. Project Structure

```
alerts_notifier/                  (repo root; product name: AlertNotifier)
├── README.md
├── PLAN.md
├── requirements.txt
├── config/
│   ├── scoring_config.yaml
│   └── escalation_matrix.yaml
├── data/generated/               # synthetic dataset (CSV/Parquet)
├── src/alertnotifier/
│   ├── models.py                 # Pydantic Event schema (domain-agnostic)
│   ├── db.py                     # SQLAlchemy + SQLite
│   ├── sources/                  # pluggable event sources
│   │   ├── synthetic.py          # generic labelled generator
│   │   └── csv_source.py         # bring-your-own-data (later)
│   ├── ingestion/
│   │   ├── api.py                # FastAPI POST /events
│   │   └── replay.py             # replay harness (live demo)
│   ├── detectors/
│   │   ├── rules.py              # the generic rule detectors
│   │   └── ml_anomaly.py         # IsolationForest
│   ├── scoring/engine.py         # score_event()
│   ├── correlation/incidents.py  # dedup + grouping + lifecycle
│   ├── escalation/matrix.py
│   ├── audit/log.py              # append-only, hash-chained
│   └── evaluation/metrics.py     # precision/recall/FP, PR curves
├── scripts/
│   ├── generate_events.py        # produce the synthetic dataset
│   └── run_replay.py
├── dashboard/app.py              # Streamlit
├── tests/
└── docs/
    ├── architecture.md
    └── detector_cards/           # one spec card per detector
```

---

## 6. Build Phases (~8–9 days of focused work, no deadline)

**Phase 1 — Schema & generic synthetic data.** ✅ *Done — theme: API / client abuse.*
The `Event` schema now uses a neutral `source` label (fully domain-agnostic).
`scripts/generate_events.py` produces `data/generated/api_events.{csv,parquet}`:
**entities are API clients, each event is one API request** (~21k requests, 60
clients, ~12% anomalies). It injects all six anomaly types from Section 4 at
known rates **with a ground-truth label on every one**, several deliberately
*jointly rare* (scraper signature, per-client off-hours vs. always-on clients,
sustained error drift vs. isolated errors). Per-request fields: endpoint,
status_code, latency_ms, payload_bytes, client_ip/new_ip, rolling req_count_5m &
distinct_endpoints_5m, client_tier vs. base, and per-client baseline context.

**Phase 2 — Rule detectors + evaluation harness.** ✅ *Done.*
Five rule detectors (`src/alertnotifier/detectors/rules.py`) using per-client
baselines, plus a first-class evaluation harness
(`src/alertnotifier/evaluation/metrics.py`) reporting precision / recall /
false-positive rate per detector and a PR curve — all measured against the
labels. Headline results: off_baseline & status_change ~100%; value_spike 99%
recall / 66% precision (the elevated-latency noise); rate_anomaly 95% precision;
**the rules catch 0% of the scraper (`multivariate_outlier`) — the exact gap
Phase 3's ML must close.** A naive global night-hours rule scores 8% precision
vs. 100% for the per-client off_baseline rule — the case for baselines, in
numbers. Run: `python scripts/run_detectors.py`. 7 passing pytest cases.

**Phase 3 — Statistical / ML layer.** ✅ *Done.*
IsolationForest (`src/alertnotifier/detectors/ml_anomaly.py`) fit unsupervised on
7 engineered features. Measured against the held-out scraper the rules got 0% on:
**it recovers 99.7% (729/731), ranking AP 0.82.** Per-type recovery shows the
model and rules are *complementary* — IF is strong on the scraper and rate/drift
but weak (~12%) on value_spike (it's global; the per-client rule wins there) and
near-zero on off_baseline/status_change (no features for them, by design). The
payoff is the combined system: rules-only recall 55% → **rules + IF 90%**
(F1 0.67 → 0.82) at a modest precision cost (85% → 76%). Run:
`python scripts/run_ml.py`. 9 passing tests.

**Phase 4 — Risk scoring engine.** ✅ *Done.*
`RiskScorer` (`src/alertnotifier/scoring/engine.py`) blends four components —
detection severity, the source severity hint, persistence (sustained across the
entity's recent events), and recurrence (repeat offender) — then scales by an
entity-criticality multiplier, all weighted from `config/scoring_config.yaml`,
into a 0–100 score. Result: **normal traffic medians 16, anomalies 48–80**
(scraper highest at 80), clean separation (single-score average precision 0.77).
The high band [75–100] holds 1042 anomalies vs 186 normal — the dial Phase 5
escalates on. Run: `python scripts/run_scoring.py`. 13 passing tests.

**Phase 5 — Incident correlation + escalation matrix.**
Correlation layer: dedup repeat alerts, group related alerts by entity + time
window into a single incident with a running score and lifecycle
(`open → escalated → resolved`). Escalate *incidents*, not raw alerts.
`escalation_matrix.yaml`: score bands → `log_only / notify / notify+SLA /
auto_contain`. Auto-actions are stubbed (log "would page on-call" / hit a mock
webhook / mark contained).

**Phase 6 — Audit log + replay harness.**
Append-only `audit_log`: event, detector, score, decision, action, timestamps.
**Hash-chain each row** (stores the previous row's hash) for tamper-evidence,
plus a verifier that detects tampering. **Replay harness**: stream the dataset
through the full engine at adjustable speed so the pipeline runs live.

**Phase 7 — Dashboard.**
Streamlit: live feed (driven by replay), score distribution, top recurring
incidents, an **incident view** with drill-down to the underlying events, and an
auto-generated summary of the monitoring window. Replay + live incidents = the
demo moment.

**Phase 8 — Polish & packaging.**
README with the architecture diagram, setup steps, screenshots, and the eval
numbers up front. **Detector spec cards** in `docs/detector_cards/` (per
detector: what it catches, what it misses, false-positive sources, why this
threshold). Wire up `POST /events`. Clean commits, push to GitHub. Optional:
short screen recording.

---

## 7. Deliverables Checklist

- [ ] GitHub repo with README + architecture diagram
- [x] One generic synthetic dataset **with ground-truth labels** — API/client-abuse theme (~21k requests)
- [x] Rule detectors for the generic anomaly taxonomy, with passing tests
- [x] **Evaluation harness: precision / recall / FP-rate per detector + PR curve**
- [x] IsolationForest layer **measured against held-out anomalies** — 99.7% scraper recovery (rules: 0%)
- [x] Config-driven scoring (`scoring_config.yaml`) — normal median 16, anomalies 48–80
- [ ] **Incident correlation + dedup with lifecycle**
- [ ] Escalation matrix (`escalation_matrix.yaml`) with (stubbed) auto-actions
- [ ] Hash-chained audit log + tamper verifier
- [ ] **Replay harness driving a live dashboard**
- [ ] Dashboard with incident view + window summary
- [ ] Detector spec cards (`docs/detector_cards/`)
- [ ] (Stretch) A `csv_source` so the engine can run on real data you bring

---

## 8. Notes & Limitations

The default dataset is **synthetic** — state that plainly in the README. What
the project demonstrates is the engineering: a clean domain-agnostic pipeline,
detection you can *measure* (real precision/recall because the data is
labelled), incident correlation, tamper-evident auditing, and a live dashboard.
Because ingestion is source-agnostic, swapping the synthetic generator for a
real CSV or stream is a small, well-isolated change — a natural next step and a
good way to show the engine works beyond its test data.
