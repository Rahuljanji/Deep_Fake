
import streamlit as st
import numpy as np
import librosa
import tempfile
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Executive profile store (in production: pull from a secure database)
# ---------------------------------------------------------------------------
EXEC_PROFILES = {
    "CEO": {
        "name": "Jane Smith",
        "voiceprint": None,          # np.ndarray of enrolled MFCC features
        "typical_vocab": ["strategy", "board", "quarterly", "shareholders"],
        "never_requests": ["wire transfer by EOD", "keep this secret", "bypass approval"],
        "usual_channels": ["email", "Teams"],
    },
    "CFO": {
        "name": "Raj Patel",
        "voiceprint": None,
        "typical_vocab": ["budget", "forecast", "variance", "EBITDA"],
        "never_requests": ["urgent wire", "personal account", "skip verification"],
        "usual_channels": ["email", "Slack"],
    },
}

@dataclass
class ImpersonationResult:
    identity_score: float = 0.0       # 0–100, how well voice matches enrolled profile
    behaviour_score: float = 0.0      # 0–100, deviation from known speech patterns
    context_flags: list = field(default_factory=list)
    overall_risk: str = "Unknown"
    overall_score: float = 0.0
    recommendation: str = ""


class ExecImpersonationDetector:

    def verify_voice_identity(
        self, audio_path: str, exec_key: str
    ) -> float:
        """
        Compare incoming voice MFCC features against enrolled voiceprint.
        Returns similarity score 0–100 (100 = perfect match).
        In production: use a dedicated speaker verification model
        (e.g. SpeechBrain ECAPA-TDNN, Resemblyzer, or Azure Speaker Recognition).
        """
        profile = EXEC_PROFILES.get(exec_key)
        if not profile or profile["voiceprint"] is None:
            return 50.0  # no baseline enrolled — neutral score

        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).mean(axis=1)
        enrolled = np.array(profile["voiceprint"])

        cosine_sim = np.dot(mfcc, enrolled) / (
            np.linalg.norm(mfcc) * np.linalg.norm(enrolled) + 1e-8
        )
        return round(float(np.clip(cosine_sim * 100, 0, 100)), 1)

    def analyse_speech_behaviour(
        self, transcript: str, exec_key: str
    ) -> tuple[float, list]:
        """
        Check vocabulary and phrasing against the executive's known baseline.
        Returns (anomaly_score, matched_flags).
        """
        profile = EXEC_PROFILES.get(exec_key, {})
        red_phrases = profile.get("never_requests", [])
        typical_vocab = profile.get("typical_vocab", [])
        text_lower = transcript.lower()

        flags = []

        # Hard red flags — phrases this exec would never say
        for phrase in red_phrases:
            if phrase.lower() in text_lower:
                flags.append(f'Unusual phrase: "{phrase}"')

        # Urgency language patterns
        urgency_words = ["immediately", "right now", "today only", "urgent",
                         "no time", "asap", "before close"]
        secrecy_words = ["don't tell", "between us", "confidential", "no one else",
                         "bypass", "skip", "without approval"]

        urgency_count = sum(1 for w in urgency_words if w in text_lower)
        secrecy_count = sum(1 for w in secrecy_words if w in text_lower)

        if urgency_count >= 2:
            flags.append(f"High urgency language ({urgency_count} instances)")
        if secrecy_count >= 1:
            flags.append(f"Secrecy/bypass language ({secrecy_count} instances)")

        # Vocabulary drift — how much of the message matches known vocab
        vocab_hits = sum(1 for v in typical_vocab if v in text_lower)
        vocab_score = vocab_hits / max(len(typical_vocab), 1)

        anomaly_score = min(100, len(flags) * 30 + (1 - vocab_score) * 40)
        return round(anomaly_score, 1), flags

    def check_context_risk(
        self,
        channel: str,
        request_type: str,
        exec_key: str,
    ) -> tuple[float, list]:
        """
        Evaluate whether the channel and request type are consistent
        with this executive's normal patterns.
        """
        profile = EXEC_PROFILES.get(exec_key, {})
        usual_channels = profile.get("usual_channels", [])
        flags = []

        if channel not in usual_channels:
            flags.append(f"Unusual channel: {channel} (normally uses {', '.join(usual_channels)})")

        HIGH_RISK_REQUESTS = [
            "wire transfer", "bank transfer", "gift cards",
            "crypto", "override approval", "vendor payment",
        ]
        for hr in HIGH_RISK_REQUESTS:
            if hr.lower() in request_type.lower():
                flags.append(f"High-risk request type: {request_type}")
                break

        risk_score = min(100, len(flags) * 45)
        return round(risk_score, 1), flags

    def aggregate(
        self,
        identity_score: float,
        behaviour_score: float,
        context_score: float,
        context_flags: list,
    ) -> ImpersonationResult:
        # Lower identity similarity = more suspicious
        identity_risk = 100 - identity_score

        overall = (identity_risk * 0.4 + behaviour_score * 0.35 + context_score * 0.25)

        if overall >= 70:
            risk, rec = "High", "Block request. Alert security team immediately."
        elif overall >= 40:
            risk, rec = "Medium", "Require secondary verification (call-back on known number)."
        else:
            risk, rec = "Low", "Request appears legitimate. Proceed with standard approvals."

        return ImpersonationResult(
            identity_score=identity_score,
            behaviour_score=behaviour_score,
            context_flags=context_flags,
            overall_risk=risk,
            overall_score=round(overall, 1),
            recommendation=rec,
        )


def run():
    st.title("Executive / C-Suite Impersonation Detection")
    st.markdown("Detect fraudulent communications impersonating senior leadership.")

    exec_key = st.selectbox("Claimed executive", list(EXEC_PROFILES.keys()))
    channel = st.selectbox("Communication channel", ["WhatsApp", "Phone call", "email", "Teams", "Zoom", "SMS"])
    request_type = st.text_input("Request being made", placeholder="e.g. wire transfer of $50,000 to vendor")
    transcript = st.text_area(
        "Message / call transcript",
        placeholder="Paste the transcript or message content here...",
        height=140,
    )
    audio_file = st.file_uploader("Upload audio (optional)", type=["wav", "mp3", "m4a"])

    if not st.button("Analyse for impersonation"):
        return

    detector = ExecImpersonationDetector()
    with st.spinner("Running impersonation checks..."):

        # Voice identity
        identity_score = 50.0
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                identity_score = detector.verify_voice_identity(tmp.name, exec_key)

        # Behaviour analysis
        behaviour_score, behaviour_flags = detector.analyse_speech_behaviour(transcript, exec_key)

        # Context risk
        context_score, context_flags = detector.check_context_risk(channel, request_type, exec_key)

        all_flags = behaviour_flags + context_flags
        result = detector.aggregate(identity_score, behaviour_score, context_score, all_flags)

    # Display result
    if result.overall_risk == "High":
        st.error(f"HIGH RISK — Likely impersonation attempt ({result.overall_score}%)")
    elif result.overall_risk == "Medium":
        st.warning(f"MEDIUM RISK — Verify before acting ({result.overall_score}%)")
    else:
        st.success(f"LOW RISK — Communication appears legitimate ({result.overall_score}%)")

    st.info(f"Recommendation: {result.recommendation}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Voice match", f"{result.identity_score}%", help="100% = perfect match to enrolled voiceprint")
    col2.metric("Behaviour anomaly", f"{result.behaviour_score}%")
    col3.metric("Context risk", f"{context_score}%")

    if result.context_flags:
        st.markdown("**Red flags detected**")
        for flag in result.context_flags:
            st.warning(flag)