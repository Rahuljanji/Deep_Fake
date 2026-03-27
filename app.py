import streamlit as st

st.set_page_config(
    page_title="DeepGuard — Detection Suite",
    page_icon="🛡️",
    layout="wide",
)

st.sidebar.title("🛡️ DeepGuard")
st.sidebar.markdown("**v2.1 · Detection Suite**")

page = st.sidebar.radio(
    "Module",
    ["🎭 Face Detection", "🎙️ Voice Spoof", "👁️ Liveness Check", "🧬 Biometric Spoof","🕴️ Exec Impersonation", "📊 Risk Dashboard"]
)

if page == "🎭 Face Detection":
    from modules.face_detection import run
    run()
elif page == "🎙️ Voice Spoof":
    from modules.voice_spoof import run
    run()
elif page == "👁️ Liveness Check":
    from modules.liveness import run
    run()
elif page == "🧬 Biometric Spoof":
    from modules.biometric_spoof import run
    run()
elif page == "🕴️ Exec Impersonation":
    from modules.exec_impersonation import run
    run()
elif page == "📊 Risk Dashboard":
    from modules.risk_dashboard import run
    run()