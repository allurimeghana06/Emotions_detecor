import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, Flatten, Dense
)


st.set_page_config(
    page_title="Emotion Recognition Model",
    layout="centered"
)


st.markdown("""
<style>
h1, h3, p {
    text-align: center;
}

.stImage > img {
    border-radius: 20px;
    border: 4px solid #3b82f6;
    box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.35);
}

div.stButton > button {
    width: 260px;
    height: 52px;
    font-weight: bold;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


st.title("Emotion Detector")
st.markdown("""
Real-time facial emotion recognition system powered by a deep convolutional
neural network. The model analyzes facial expressions from a live webcam feed
and classifies emotions with confidence scores directly on the video stream.
""")
st.divider()


@st.cache_resource
def load_emotion_model():
    model = Sequential([
        Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(512, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])

    model.load_weights("best_model.h5")
    return model

model = load_emotion_model()

# --- 5. FACE DETECTOR ---
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

emotion_labels = ["Angry", "Happy", "Neutral", "Sad"]

# --- 6. SESSION STATE ---
if "active" not in st.session_state:
    st.session_state.active = False

def toggle_camera():
    st.session_state.active = not st.session_state.active

# --- 7. PERFECTLY CENTERED BUTTON ---
left, center, right = st.columns([1, 1, 1])
with center:
    btn_label = "DEACTIVATE CAMERA" if st.session_state.active else "ACTIVATE CAMERA"
    st.button(btn_label, on_click=toggle_camera, type="primary")

FRAME_WINDOW = st.image([])

# --- 8. CAMERA LOOP ---
if st.session_state.active:
    cap = cv2.VideoCapture(0)

    while st.session_state.active:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

            preds = model.predict(roi, verbose=0)[0]
            idx = np.argmax(preds)
            emotion = emotion_labels[idx]
            confidence = preds[idx] * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (59, 130, 246), 3)

            cv2.putText(
                frame, emotion,
                (x, y - 35),
                cv2.FONT_HERSHEY_DUPLEX,
                1.1, (59, 130, 246), 2
            )

            cv2.putText(
                frame, f"{confidence:.1f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2
            )

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

# --- 9. CAMERA OFF ---
else:
    FRAME_WINDOW.empty()




