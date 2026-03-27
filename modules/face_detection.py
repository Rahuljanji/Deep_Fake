import streamlit as st
from PIL import Image
import numpy as np
import cv2
from models.deepfake_model import DeepfakeDetector

@st.cache_resource
def load_detector():
    return DeepfakeDetector()  # pass model_path if you have weights

def run():
    st.title("🎭 Face Deepfake Detection")
    st.markdown("Upload an image or video to detect synthetic faces.")

    tab1, tab2 = st.tabs(["📸 Image", "🎬 Video"])

    with tab1:
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Analyze Image"):
                with st.spinner("Running deepfake detection..."):
                    detector = load_detector()
                    result = detector.predict(image)

                _display_result(result)

    with tab2:
        uploaded_vid = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
        if uploaded_vid and st.button("🔍 Analyze Video"):
            _analyze_video(uploaded_vid)


def _display_result(result: dict):
    is_fake = result["prediction"] == "FAKE"
    col1, col2, col3 = st.columns(3)

    if is_fake:
        st.error(f"❌ **Deepfake Detected** — Confidence: {result['fake_probability']}%")
    else:
        st.success(f"✅ **Authentic Face** — Confidence: {result['real_probability']}%")

    col1.metric("Fake Probability", f"{result['fake_probability']}%")
    col2.metric("Real Probability", f"{result['real_probability']}%")
    risk = "🔴 High" if result['fake_probability'] > 70 else "🟡 Medium" if result['fake_probability'] > 40 else "🟢 Low"
    col3.metric("Risk Level", risk)

    # Confidence bar
    st.progress(result["fake_probability"] / 100, text="Fake probability")


def _analyze_video(video_file):
    import tempfile, os
    detector = load_detector()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fake_scores = []
    progress = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, frame_count, 15):  # sample every 15 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        r = detector.predict(pil_img)
        fake_scores.append(r["fake_probability"])
        progress.progress(min(i / frame_count, 1.0))

    cap.release()
    os.unlink(tmp_path)

    avg = np.mean(fake_scores)
    st.metric("Average Fake Score", f"{avg:.1f}%")
    st.line_chart(fake_scores)