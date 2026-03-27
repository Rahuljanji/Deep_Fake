import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
BLINK_THRESHOLD = 0.22
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158,  133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def run():
    st.title("👁️ Liveness Detection")
    st.markdown("Uses your webcam to verify blink, head movement, and facial landmarks.")

    FRAME_WINDOW = st.image([])
    status_box = st.empty()
    blink_counter = st.empty()

    run_camera = st.toggle("Start Camera", value=False)

    blink_count = 0
    blink_flag = False

    if run_camera:
        cap = cv2.VideoCapture(0)
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            stop_btn = st.button("⏹ Stop")
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
                    right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                    avg_ear   = (left_ear + right_ear) / 2

                    if avg_ear < BLINK_THRESHOLD:
                        blink_flag = True
                    elif blink_flag:
                        blink_count += 1
                        blink_flag = False

                    live = blink_count >= 2
                    label = "✅ LIVE" if live else "⏳ Checking..."
                    color = (0, 200, 100) if live else (255, 180, 0)
                    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                blink_counter.metric("Blinks Detected", blink_count)

        cap.release()