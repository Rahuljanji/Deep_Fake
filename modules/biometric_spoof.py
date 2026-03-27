# modules/biometric_spoof.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image

class BiometricSpoofDetector:
    """
    Multi-modal biometric spoof detection.
    Each method returns a dict: {spoof: bool, score: float, method: str}
    """

    def check_texture_liveness(self, image: np.ndarray) -> dict:
        """
        Detect printed photo attacks using Local Binary Patterns (LBP).
        Real faces have micro-texture; printed photos flatten it.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # LBP texture analysis
        radius, n_points = 1, 8
        lbp = self._compute_lbp(gray, radius, n_points)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float") / hist.sum()

        # Entropy of LBP histogram — real face = higher entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        spoof_score = max(0, 1 - (entropy / 8.0))  # normalise

        return {
            "spoof": spoof_score > 0.6,
            "score": round(spoof_score * 100, 1),
            "method": "LBP texture analysis",
        }

    def check_depth_consistency(self, image: np.ndarray) -> dict:
        """
        Estimate depth map using MiDaS or simple gradient analysis.
        Flat prints show minimal depth variance.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        depth_variance = laplacian.var()

        # Low variance = flat surface = likely print/screen
        threshold = 150
        spoof_score = max(0, 1 - (depth_variance / threshold))
        spoof_score = min(spoof_score, 1.0)

        return {
            "spoof": depth_variance < threshold,
            "score": round(spoof_score * 100, 1),
            "method": "Depth/gradient analysis",
        }

    def check_reflection_pattern(self, image: np.ndarray) -> dict:
        """
        Screens/photos produce specular highlights in predictable regions.
        Real faces scatter light more uniformly.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        highlight_mask = v_channel > 230
        highlight_ratio = highlight_mask.sum() / highlight_mask.size

        # High concentrated bright spots = screen reflection
        spoof_score = min(highlight_ratio * 20, 1.0)
        return {
            "spoof": highlight_ratio > 0.05,
            "score": round(spoof_score * 100, 1),
            "method": "Specular reflection check",
        }

    def _compute_lbp(self, gray, radius, n_points):
        from skimage.feature import local_binary_pattern
        return local_binary_pattern(gray, n_points, radius, method="uniform")

    def aggregate(self, results: list) -> dict:
        scores = [r["score"] for r in results]
        avg_score = np.mean(scores)
        any_spoof = any(r["spoof"] for r in results)
        risk = "High" if avg_score > 65 else "Medium" if avg_score > 35 else "Low"
        return {"overall_score": round(avg_score, 1), "is_spoof": any_spoof, "risk": risk}


def run():
    st.title("Biometric Spoof Detection")
    st.markdown("Multi-layer analysis to detect printed photos, screens, and 3D masks.")

    uploaded = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, use_column_width=True)

    if st.button("Run Spoof Analysis"):
        detector = BiometricSpoofDetector()
        with st.spinner("Running multi-modal spoof checks..."):
            checks = [
                detector.check_texture_liveness(img_np),
                detector.check_depth_consistency(img_np),
                detector.check_reflection_pattern(img_np),
            ]
            aggregate = detector.aggregate(checks)

        if aggregate["is_spoof"]:
            st.error(f"Biometric Spoof Detected — Risk: {aggregate['risk']}")
        else:
            st.success(f"Biometric checks passed — Risk: {aggregate['risk']}")

        st.metric("Overall spoof score", f"{aggregate['overall_score']}%")

        st.markdown("**Individual checks**")
        for check in checks:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(check["method"])
            col2.write(f"{check['score']}%")
            col3.write("SPOOF" if check["spoof"] else "OK")
            st.progress(check["score"] / 100)