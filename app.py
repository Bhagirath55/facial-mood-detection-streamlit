import streamlit as st
import cv2
import numpy as np
import pickle
import os

# Page Config
st.set_page_config(page_title="Facial Mood Detection", layout="centered")

# Loading Model & Labels
svm = cv2.ml.SVM_load("emotion_svm.xml")

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# Loading Haar Cascade (robust path)
HAAR_PATH = os.path.join(
    os.path.dirname(cv2.__file__),
    "data",
    "haarcascade_frontalface_default.xml"
)
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

# HOG Descriptor
hog = cv2.HOGDescriptor(
    _winSize=(48, 48),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)


# Preprocessing Function
def preprocess_bgr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )

    # Use detected face if available, else full image
    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = gray[y:y + h, x:x + w]
    else:
        roi = gray

    roi = cv2.resize(roi, (48, 48))
    roi = roi / 255.0

    hog_feat = np.array(
        hog.compute((roi * 255).astype(np.uint8))
    ).flatten()

    return hog_feat.astype(np.float32)


# UI
st.title("Facial Mood Detection System")

mode = st.radio(
    "Select Input Mode",
    [
        "Upload Image",
        "Live Camera (Snapshot)",
        "Live Camera (Real-Time)"
    ]
)

# MODE 1: IMAGE UPLOAD
if mode == "Upload Image":
    uploaded = st.file_uploader(
        "Upload a facial image", type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(data, 1)

        features_vec = preprocess_bgr(img)
        _, pred = svm.predict(features_vec.reshape(1, -1))
        emotion = inv_label_map[int(pred[0][0])]

        st.image(img, channels="BGR")
        st.success(f"Predicted Mood: {emotion}")

# MODE 2: CAMERA SNAPSHOT
elif mode == "Live Camera (Snapshot)":
    frame = st.camera_input("Capture image from camera")

    if frame:
        data = np.asarray(bytearray(frame.read()), dtype=np.uint8)
        img = cv2.imdecode(data, 1)

        features_vec = preprocess_bgr(img)
        _, pred = svm.predict(features_vec.reshape(1, -1))
        emotion = inv_label_map[int(pred[0][0])]

        st.image(img, channels="BGR")
        st.success(f"Predicted Mood: {emotion}")

# MODE 3: REAL-TIME CAMERA
elif mode == "Live Camera (Real-Time)":

    st.warning("This mode works only when run locally (not on Streamlit Cloud).")

    run = st.checkbox("START CAMERA")

    FRAME_WINDOW = st.image([])
    status_text = st.empty()

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            status_text.error("Unable to access camera.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    status_text.error("Failed to read from camera.")
                    break

                features_vec = preprocess_bgr(frame)
                _, pred = svm.predict(features_vec.reshape(1, -1))
                emotion = inv_label_map[int(pred[0][0])]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)
                status_text.success(f"Live Mood: {emotion}")

            cap.release()
