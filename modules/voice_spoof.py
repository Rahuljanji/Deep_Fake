import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import tempfile

def extract_features(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.hstack([
        mfcc.mean(axis=1),
        chroma.mean(axis=1),
        spectral_contrast.mean(axis=1),
    ])

def mock_voice_predict(features: np.ndarray) -> dict:
    # Replace with your trained voice spoof classifier
    score = float(np.random.uniform(0.1, 0.95))
    return {
        "prediction": "SPOOF" if score > 0.5 else "GENUINE",
        "spoof_probability": round(score * 100, 1),
        "genuine_probability": round((1 - score) * 100, 1),
    }

def run():
    st.title("🎙️ Voice Spoof Detection")
    st.markdown("Upload an audio file to detect synthetic or replayed voice.")

    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg"])
    if not uploaded:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    if st.button("🔍 Analyze Voice"):
        with st.spinner("Extracting audio features..."):
            features = extract_features(tmp_path)
            result = mock_voice_predict(features)

        is_spoof = result["prediction"] == "SPOOF"
        if is_spoof:
            st.error(f"❌ **Synthetic Voice Detected** — {result['spoof_probability']}% confidence")
        else:
            st.success(f"✅ **Genuine Voice** — {result['genuine_probability']}% confidence")

        col1, col2 = st.columns(2)
        col1.metric("Spoof Score", f"{result['spoof_probability']}%")
        col2.metric("Genuine Score", f"{result['genuine_probability']}%")
        st.progress(result["spoof_probability"] / 100, text="Spoof probability")