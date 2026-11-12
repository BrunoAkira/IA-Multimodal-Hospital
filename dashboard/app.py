"""
Painel local (Streamlit) que lê os arquivos gerados pelo pipeline em results/.

Páginas: Overview | Alerts | Patient timeline | Clinical | Voice & text | Physio video | Live monitor
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config                                  # noqa: E402
from src.clinical.vitals import VitalStreamDetector, VITALS  # noqa: E402

R = config.RESULTS_DIR
st.set_page_config(page_title="Hospital AI", page_icon="🏥", layout="wide")


# ---------------------------------------------------------------------------
# Leitura dos resultados (com cache do Streamlit para não reler a cada clique)
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(name, **kw):
    p = R / name
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


@st.cache_data
def load_json(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else {}


alerts = load_csv("alerts.csv", parse_dates=["timestamp"])
events = load_csv("events_fused.csv", parse_dates=["timestamp"])
vp = load_csv("virtual_patients.csv")

st.sidebar.title("🏥 Hospital AI")
st.sidebar.caption("Research prototype — not for clinical decisions")
page = st.sidebar.radio("Page", ["Overview", "Alerts", "Patient timeline", "Clinical",
                                 "Voice & text", "Physio video", "Live monitor"])
if st.sidebar.button("Reload results"):
    st.cache_data.clear()
    st.rerun()

LEVEL_COLORS = {"critical": "#b00020", "high": "#e65100", "medium": "#f9a825"}

# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Multimodal patient monitoring")
    if events.empty:
        st.warning("No results yet. Run `python main.py all` first.")
        st.stop()
    c = st.columns(5)
    c[0].metric("Patients monitored", events["patient_id"].nunique())
    c[1].metric("Virtual patients (fused)", len(vp))
    c[2].metric("Events", len(events))
    c[3].metric("Alerts", len(alerts))
    c[4].metric("Critical alerts", int((alerts["level"] == "critical").sum()) if len(alerts) else 0)

    left, right = st.columns(2)
    by_mod = events.groupby(["modality", "severity"]).size().reset_index(name="count")
    left.plotly_chart(px.bar(by_mod, x="modality", y="count", color="severity",
                             title="Events by modality and severity"))
    top = events["event_type"].value_counts().head(15).reset_index()
    top.columns = ["event_type", "count"]
    right.plotly_chart(px.bar(top, x="count", y="event_type", orientation="h", title="Most frequent event types"),
                       )
    usage = load_json("audio_summary.json").get("azure_usage", {})
    if usage:
        st.subheader("Azure free-tier usage (this month)")
        u1, u2 = st.columns(2)
        u1.progress(min(1.0, usage["speech_seconds"] / usage["speech_budget_seconds"]),
                    text=f"Speech: {usage['speech_seconds'] / 60:.1f} / {usage['speech_budget_seconds'] / 60:.0f} min")
        u2.progress(min(1.0, usage["language_records"] / usage["language_budget_records"]),
                    text=f"Language: {usage['language_records']} / {usage['language_budget_records']} records")

# ---------------------------------------------------------------------------
elif page == "Alerts":
    st.title("Alert queue")
    if alerts.empty:
        st.info("No alerts.")
        st.stop()
    levels = st.multiselect("Level", ["critical", "high", "medium"], default=["critical", "high"])
    multi = st.checkbox("Only multimodal alerts (2+ modalities)")
    a = alerts[alerts["level"].isin(levels)]
    if multi:
        a = a[a["n_strong_modalities"] >= 2]
    st.caption(f"{len(a)} alerts")
    st.dataframe(a.sort_values(["timestamp"], ascending=False), hide_index=True)

# ---------------------------------------------------------------------------
elif page == "Patient timeline":
    st.title("Patient timeline")
    if events.empty:
        st.stop()
    pids = sorted(events["patient_id"].unique(), key=lambda x: (not x.startswith("VP"), x))
    pid = st.selectbox("Patient", pids)
    if pid.startswith("VP") and not vp.empty:
        st.caption("Virtual patient = " + " + ".join(f"{k}: {v}" for k, v in
                                                     vp[vp["virtual_id"] == pid].iloc[0].items()
                                                     if k != "virtual_id" and pd.notna(v)))
    pe = events[events["patient_id"] == pid]
    fig = px.scatter(pe, x="timestamp", y="modality", color="severity", size="score", hover_data=["event_type", "message"],
                     color_discrete_map={"high": "#b00020", "medium": "#f9a825", "low": "#90a4ae", "info": "#cfd8dc"},
                     title="Events by modality")
    for r in alerts[alerts["patient_id"] == pid].itertuples():
        fig.add_vline(x=r.timestamp, line_dash="dot", line_color=LEVEL_COLORS[r.level])
    st.plotly_chart(fig)

    # sinais vitais do paciente do MIMIC por trás do paciente virtual
    hourly = load_csv("vitals_hourly.csv", parse_dates=["charttime"])
    src = vp.loc[vp["virtual_id"] == pid, "mimic"].iloc[0] if pid.startswith("VP") else pid
    if src.startswith("mimic:") and not hourly.empty:
        h = hourly[hourly["subject_id"] == int(src.split(":")[1])]
        vit = st.multiselect("Vitals", VITALS, default=["heart_rate", "systolic_bp", "spo2", "resp_rate"])
        fig2 = go.Figure()
        for v in vit:
            if v in h:
                fig2.add_trace(go.Scatter(x=h["charttime"], y=h[v], name=v, mode="lines"))
        fig2.update_layout(title="Hourly vital signs", height=380)
        st.plotly_chart(fig2)
    st.dataframe(pe[["timestamp", "modality", "source", "event_type", "severity", "score", "message"]],
                 hide_index=True)

# ---------------------------------------------------------------------------
elif page == "Clinical":
    st.title("Clinical anomalies (MIMIC-IV Demo)")
    s = load_json("clinical_summary.json")
    if not s:
        st.warning("Run `python main.py clinical`.")
        st.stop()
    st.json(s["events"])
    if "injection_evaluation" in s:
        st.subheader("Detector evaluation with injected anomalies")
        st.dataframe(pd.DataFrame(s["injection_evaluation"]).T)
        st.caption("Precision is conservative: 'false positives' include real anomalies in MIMIC.")
    ce = events[events["modality"] == "clinical"]
    st.plotly_chart(px.histogram(ce, x="event_type", color="source", title="Clinical events"),
                    )

# ---------------------------------------------------------------------------
elif page == "Voice & text":
    st.title("Voice analysis (TORGO) and consultations")
    s = load_json("audio_summary.json")
    if not s:
        st.warning("Run `python main.py audio`.")
        st.stop()
    c = st.columns(3)
    c[0].metric("Utterances", s["utterances"])
    c[1].metric("Speakers", s["speakers"])
    c[2].metric("Consultations", s["consultations"])
    clf = s.get("classifier", {})
    if "metrics" in clf:
        st.subheader(f"Dysarthria classifier — validated on unseen speakers (best: {clf['best_model']})")
        st.dataframe(pd.DataFrame(clf["metrics"]).T.drop(columns="confusion_matrix"))
    st.subheader("Azure Speech to Text — word error rate by group")
    st.write(s.get("azure_stt_wer_by_group"))
    feats = load_csv("voice_features.csv")
    if not feats.empty:
        f = st.selectbox("Voice feature", ["speech_rate_sps", "pause_ratio", "long_pauses", "hnr_db",
                                           "jitter", "shimmer", "f0_std_st", "rms_cv"])
        st.plotly_chart(px.box(feats, x="speaker_id", y=f, color="group", title=f"{f} by speaker"),
                        )
    cons = load_json("consultations.json")
    if cons:
        st.subheader("Consultations — critical terms and sentiment")
        st.dataframe(pd.DataFrame(cons)[["consultation_id", "transcript_source", "sentiment",
                                         "sentiment_source", "critical_terms", "negated_terms"]],
                     hide_index=True)

# ---------------------------------------------------------------------------
elif page == "Physio video":
    st.title("Physiotherapy movement analysis")
    s = load_json("video_summary.json")
    if not s:
        st.warning("Run `python main.py video`.")
        st.stop()
    if "ulred" in s:
        st.subheader("UL-RED repetitions vs. reference")
        st.json(s["ulred"])
        reps = load_csv("physio_repetitions.csv")
        if not reps.empty:
            st.plotly_chart(px.scatter(reps, x="duration_ratio", y="rom_ratio", color="pace_label",
                                       size="anomaly_score", hover_data=["subject", "motion", "rep"],
                                       title="Each dot = one repetition (fast/slow should stand out)"),
                            )
    for v in s.get("videos", []):
        st.subheader(f"Video: {v['video']}")
        st.json(v)
    reports = sorted(config.REPORTS_DIR.glob("*.html"))
    if reports:
        st.subheader("Automatic reports")
        for rp in reports:
            st.download_button(f"Download {rp.name}", rp.read_bytes(), file_name=rp.name, mime="text/html")

# ---------------------------------------------------------------------------
elif page == "Live monitor":
    st.title("Live monitor (real-time replay)")
    st.caption("Replays a patient's hourly vitals as if they were arriving from the bedside monitor.")
    hourly = load_csv("vitals_hourly.csv", parse_dates=["charttime"])
    if hourly.empty:
        st.warning("Run `python main.py clinical`.")
        st.stop()
    subj = st.selectbox("MIMIC patient", sorted(hourly["subject_id"].unique()))
    speed = st.slider("Seconds per simulated hour", 0.05, 1.0, 0.2)
    if st.button("▶ Start"):
        h = hourly[hourly["subject_id"] == subj].sort_values("charttime")
        det = VitalStreamDetector()
        chart, feed = st.empty(), st.empty()
        seen, log = [], []
        for r in h.to_dict("records"):
            seen.append(r)
            for v in VITALS:
                if pd.notna(r.get(v)):
                    score, etype, _ = det.update(r["stay_id"], v, float(r[v]))
                    if score >= 0.5:
                        log.insert(0, f"{'🔴' if score >= 0.8 else '🟠'} {r['charttime']}  {etype}  ({v}={r[v]:.1f})")
            df = pd.DataFrame(seen)
            fig = go.Figure([go.Scatter(x=df["charttime"], y=df[v], name=v, mode="lines")
                             for v in ("heart_rate", "systolic_bp", "spo2", "resp_rate") if v in df])
            fig.update_layout(height=360, margin=dict(t=20))
            chart.plotly_chart(fig, key=f"live_{len(seen)}")
            feed.markdown("\n\n".join(log[:12]) or "No alerts yet")
            time.sleep(speed)
