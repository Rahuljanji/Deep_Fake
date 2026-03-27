import streamlit as st
import plotly.graph_objects as go
import numpy as np

def run():
    st.title("📊 Risk Score Dashboard")
    st.markdown("Aggregate fraud risk across all detection modules.")

    col1, col2, col3, col4 = st.columns(4)

    # Example scores — wire in your real module outputs
    face_score  = st.session_state.get("face_score", 92)
    voice_score = st.session_state.get("voice_score", 34)
    live_score  = st.session_state.get("live_score", 15)
    overall     = int(np.mean([face_score, voice_score, live_score]))

    col1.metric("Face Score",    f"{face_score}%",  delta="High Risk",  delta_color="inverse")
    col2.metric("Voice Score",   f"{voice_score}%", delta="Medium Risk", delta_color="normal")
    col3.metric("Liveness Score",f"{live_score}%",  delta="Low Risk",   delta_color="normal")
    col4.metric("Overall Risk",  f"{overall}%")

    # Risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall,
        title={"text": "Fraud Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "crimson" if overall > 60 else "orange" if overall > 30 else "green"},
            "steps": [
                {"range": [0, 30],  "color": "#d4edda"},
                {"range": [30, 60], "color": "#fff3cd"},
                {"range": [60, 100],"color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": overall},
        },
    ))
    st.plotly_chart(fig, use_container_width=True)

    if overall > 60:
        st.error(f"🔴 **HIGH RISK** — Prediction: Deepfake Detected ❌ | Confidence: {overall}%")
    elif overall > 30:
        st.warning(f"🟡 **MEDIUM RISK** — Further review recommended | Confidence: {overall}%")
    else:
        st.success(f"🟢 **LOW RISK** — Content appears authentic ✅ | Confidence: {100-overall}%")