import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from src.utils import filter_classes, count_objects

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="YOLO AI App", layout="wide")

st.title("🚗 YOLO AI Detection Platform")

model = YOLO("yolov8n.pt")

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.header("⚙️ Settings")

classes = st.sidebar.multiselect(
    "Select objects to detect",
    ["person", "car", "bicycle", "motorbike"],
    default=["person", "car"]
)

mode = st.sidebar.radio("Mode", ["Image"])

# -----------------------
# IMAGE MODE
# -----------------------
if mode == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        start_time = time.time()

        results = model(image)

        filtered = filter_classes(results, classes)
        counts = count_objects(filtered)

        annotated = results[0].plot()

        end_time = time.time()
        fps = 1 / (end_time - start_time)

        col1, col2 = st.columns(2)

        with col1:
            st.image(annotated, channels="BGR", use_container_width=True)

        with col2:
            st.subheader("📊 Object Count")
            st.write(counts)

            st.subheader("⚡ Performance")
            st.write(f"FPS: {int(fps)}")

# -----------------------
# WEBCAM MODE
# -----------------------
elif mode == "Webcam":
    run = st.button("▶️ Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            filtered = filter_classes(results, classes)
            counts = count_objects(filtered)

            annotated = results[0].plot()

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            stframe.image(annotated)

        cap.release()