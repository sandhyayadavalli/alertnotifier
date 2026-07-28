"""AlertNotifier dashboard (Phase 7) — a live view of the whole pipeline.

Run:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alertnotifier.audit.log import AuditLog  # noqa: E402
from alertnotifier.dataset import DATA_DIR, load_dataset, split_events_labels  # noqa: E402
from alertnotifier.ingestion.replay import replay  # noqa: E402
from alertnotifier.pipeline import run_pipeline  # noqa: E402

ACTION_ORDER = ["auto_contain", "notify_security", "notify_analyst", "log_only"]
ACTION_COLOR = {"auto_contain": "#dc2626", "notify_security": "#ea580c",
                "notify_analyst": "#ca8a04", "log_only": "#2563eb"}

st.set_page_config(page_title="AlertNotifier", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner="Running detection → scoring → correlation → escalation…")
def load_data() -> dict:
    events, labels = split_events_labels(load_dataset("api_events"))
    result = run_pipeline(events)

    inc_df = pd.DataFrame([{
        "incident_id": i.incident_id, "client": i.entity_id, "risk": round(i.score, 1),
        "action": i.action, "alerts": i.alert_count, "signals": ", ".join(i.signals),
        "start": i.start, "end": i.end, "duration_min": round(i.duration_minutes, 1),
    } for i in result.incidents])

    alerts = result.alerts.copy()
    alerts["signals"] = alerts["signals"].apply(lambda s: ", ".join(s))

    idx = labels.set_index("event_id")
    risk_df = pd.DataFrame({
        "risk": result.risk.reindex(idx.index).to_numpy(),
        "kind": idx["gt_is_anomaly"].map({True: "anomaly", False: "normal"}).to_numpy(),
    })

    audit = AuditLog(":memory:")
    audit.extend({
        "incident_id": i.incident_id, "entity_id": i.entity_id, "start_ts": i.start.isoformat(),
        "end_ts": i.end.isoformat(), "alert_count": i.alert_count, "detectors": ",".join(i.signals),
        "risk_score": i.score, "action": i.action,
    } for i in replay(result.incidents))
    ok, bad = audit.verify()
    audit_df = pd.DataFrame([{
        "seq": r.seq, "incident": r.incident_id, "risk": round(r.risk_score, 1), "action": r.action,
        "prev_hash": r.prev_hash[:10] + "…", "row_hash": r.row_hash[:10] + "…",
    } for r in audit.entries()])

    return {"n_events": len(events), "n_alerts": len(alerts), "inc_df": inc_df,
            "alerts": alerts, "risk_df": risk_df, "audit_df": audit_df, "audit_ok": ok, "audit_bad": bad}


if not (DATA_DIR / "api_events.parquet").exists() and not (DATA_DIR / "api_events.csv").exists():
    st.error("No dataset found. Run `python scripts/generate_events.py` first, then reload.")
    st.stop()

data = load_data()
inc_df = data["inc_df"]

st.title("🛡️ AlertNotifier")
st.caption("Real-time risk monitoring & escalation — API / client-abuse traffic. Synthetic, labelled data.")

n_inc = len(inc_df)
n_contain = int((inc_df["action"] == "auto_contain").sum())
k = st.columns(5)
k[0].metric("Events ingested", f"{data['n_events']:,}")
k[1].metric("Alerts", f"{data['n_alerts']:,}")
k[2].metric("Incidents", f"{n_inc:,}")
k[3].metric("Auto-contained", f"{n_contain:,}")
k[4].metric("Audit chain", "✅ intact" if data["audit_ok"] else f"⚠ broken @ {data['audit_bad']}")

st.sidebar.header("Filters")
actions = st.sidebar.multiselect("Escalation action", ACTION_ORDER, default=ACTION_ORDER)
min_risk = st.sidebar.slider("Minimum risk score", 0, 100, 0)
f = inc_df[(inc_df["action"].isin(actions)) & (inc_df["risk"] >= min_risk)]

tab_overview, tab_incidents, tab_audit = st.tabs(["📊 Overview", "🚨 Incidents", "🔒 Audit trail"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Risk score distribution")
        fig = px.histogram(data["risk_df"], x="risk", color="kind", nbins=40, barmode="overlay",
                           color_discrete_map={"normal": "#2563eb", "anomaly": "#dc2626"})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Incidents by escalation action")
        counts = inc_df["action"].value_counts().reindex(ACTION_ORDER).fillna(0).reset_index()
        counts.columns = ["action", "count"]
        fig2 = px.bar(counts, x="count", y="action", orientation="h", color="action",
                      color_discrete_map=ACTION_COLOR)
        fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                           yaxis={"categoryorder": "array", "categoryarray": ACTION_ORDER[::-1]})
        st.plotly_chart(fig2, use_container_width=True)

    if n_inc:
        vc = inc_df["client"].value_counts()
        st.subheader("Window summary")
        st.markdown(
            f"- **{n_inc:,} incidents** across **{inc_df['client'].nunique()} clients**, from "
            f"{inc_df['start'].min():%Y-%m-%d %H:%M} to {inc_df['end'].max():%Y-%m-%d %H:%M}.\n"
            f"- **{n_contain}** auto-contained · "
            f"**{int((inc_df['action'] == 'notify_security').sum())}** paged to security · "
            f"**{int((inc_df['action'] == 'notify_analyst').sum())}** to an analyst.\n"
            f"- Busiest client: **{vc.idxmax()}** with **{int(vc.max())}** incidents."
        )

with tab_incidents:
    st.subheader(f"Incidents ({len(f)} shown)")
    st.dataframe(
        f.sort_values("risk", ascending=False)[
            ["incident_id", "client", "risk", "action", "alerts", "duration_min", "signals", "start"]],
        use_container_width=True, hide_index=True)

    st.subheader("Drill down")
    if len(f):
        pick = st.selectbox("Pick an incident", f.sort_values("risk", ascending=False)["incident_id"])
        inc = inc_df[inc_df["incident_id"] == pick].iloc[0]
        d = st.columns(3)
        d[0].metric("Risk", inc["risk"])
        d[1].metric("Action", inc["action"])
        d[2].metric("Alerts", inc["alerts"])
        st.write(f"**Client** {inc['client']} · **Signals** {inc['signals']} · "
                 f"**{inc['start']:%Y-%m-%d %H:%M} → {inc['end']:%H:%M}** ({inc['duration_min']} min)")
        u = data["alerts"]
        sub = u[(u["entity_id"] == inc["client"]) & (u["timestamp"] >= inc["start"]) & (u["timestamp"] <= inc["end"])]
        st.caption(f"{len(sub)} underlying alert events")
        st.dataframe(sub[["timestamp", "risk_score", "signals"]].sort_values("timestamp"),
                     use_container_width=True, hide_index=True)

with tab_audit:
    if data["audit_ok"]:
        st.success(f"Hash chain verified — {len(data['audit_df']):,} rows, tamper-evident. "
                   "Each row's hash is derived from the previous row's, so any edit is detectable.")
    else:
        st.error(f"Hash chain BROKEN at row {data['audit_bad']} — the log was tampered with.")
    st.dataframe(data["audit_df"], use_container_width=True, hide_index=True)
