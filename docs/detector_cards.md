# Detector Spec Cards

One card per detector: what it catches, what it misses, where its false positives
come from, why its threshold is set where it is, and its **measured** performance
against the labelled synthetic data (`scripts/run_detectors.py`, `run_ml.py`).

The guiding idea: rules and the ML layer are **complementary**. Each rule uses a
*per-entity baseline* where it can; the ML catches the multivariate case no single
rule expresses. Neither dominates — which is why the system keeps both.

---

## `value_spike` — ValueSpikeDetector
- **Catches:** a request whose latency or payload is extreme *relative to the
  client's own baseline* (per-client median/MAD z-score).
- **Misses:** a spike on a client with too few prior requests to form a baseline.
- **False positives:** legitimately-elevated latency (the injected ×3–5 "slow but
  fine" traffic) — the honest cost of a threshold.
- **Threshold:** robust z `> 6`. Robust (median/MAD) so the outliers we hunt don't
  inflate the baseline. Per-client so a normally-slow client isn't over-flagged.
- **Measured:** precision **66%**, recall **99%** (PR curve AP **0.97** — the
  threshold is a dial, not a fixed trade-off).

## `sustained_drift` — SustainedDriftDetector
- **Catches:** a client whose rolling error rate stays high across a window — a
  broken integration or credential-stuffing that keeps failing.
- **Misses:** the first few events of a drift (the rolling window hasn't filled).
- **False positives:** rare clusters of ordinary one-off errors.
- **Threshold:** rolling error rate `>= 0.5` over the last `8` events. Isolated
  errors (≈3% baseline) never reach it; a sustained run hits 1.0.
- **Measured:** precision **72%**, recall **78%**.

## `rate_anomaly` — RateAnomalyDetector
- **Catches:** a request-rate **burst** (scraping / DoS), or a client going
  **silent** (an expected-always-on client stops reporting).
- **Misses:** the sparse leading edge of a burst (rate not yet elevated).
- **False positives:** low — normal traffic rarely exceeds the burst threshold.
- **Threshold:** `req_count_5m > 8` for bursts; silence only for **always-on**
  clients with a gap `> 180 min`. Gating silence to always-on clients is the key
  choice — a global gap rule would false-positive on business clients overnight.
- **Measured:** precision **95%**, recall **60%**.

## `off_baseline` — OffBaselineDetector
- **Catches:** activity at an hour outside **this client's own** active window.
- **Misses:** essentially nothing of its type (the per-client flag is exact).
- **False positives:** none, *because* it's per-client.
- **Threshold:** `off_baseline_hour` (hour ∉ the client's active set). A naive
  global "night hours" rule scores **8% precision** here — it fires on every
  always-on client's legitimate 3am traffic. The per-client version scores **100%**.
  This contrast is the single clearest argument in the project for per-entity
  baselines.
- **Measured:** precision **100%**, recall **100%**.

## `status_change` — StatusChangeDetector
- **Catches:** a client whose access tier differs from its baseline tier
  (escalation or throttling).
- **Misses:** nothing of its type.
- **False positives:** negligible.
- **Threshold:** `client_tier != base_client_tier` — a state transition, no tuning.
- **Measured:** precision **99%**, recall **100%**.

## `multivariate_outlier` — IsolationForestDetector (the ML layer)
- **Catches:** the **scraper** signature — many distinct endpoints + a fresh IP +
  small payloads, all at a *normal* request rate. Every feature is normal on its
  own; only the *combination* is rare. Discovered from the joint feature space, not
  a hand-coded rule. **Held out from the rules on purpose.**
- **Misses (by design):** `value_spike` (the model is global; the per-client rule
  wins there — IF recall ~12%), and `off_baseline` / `status_change` (no features
  for them). These are the rules' job.
- **False positives:** the contamination budget flags ~10% of events, so its
  *type-specific* precision is modest; the meaningful numbers are its recall of the
  held-out type and its ranking quality.
- **Why IsolationForest:** unsupervised (no labels needed in production), built for
  rare outliers, fast, few knobs (mainly `contamination`) — every setting defensible.
- **Measured:** recovers **99.7%** of the held-out scraper the rules catch **0%** of
  (ranking AP **0.82**). Adding it lifts the whole system's recall from **55% → 90%**.

---

### The system, in one line
Rules nail the anomalies you can describe (and prove it with numbers); the
IsolationForest catches the one you can't; scoring turns both into a 0–100 dial;
correlation collapses 3,079 alerts into 1,017 incidents; escalation routes them;
and every decision lands in a tamper-evident audit log.
