# streamlit_app.py

import streamlit as st
import cv2
import tempfile
import numpy as np
import requests
import os
from PIL import Image

st.set_page_config(page_title="Real-Time DeepFake Detection", layout="centered")
st.title("🎭 Real-Time DeepFake Detection in Social Media Videos")

st.markdown("""
Upload a video, and this app will analyze it frame by frame to check for signs of DeepFakes using a pre-trained model/API.
""")

# Choose API option
api_option = st.selectbox("Choose Detection Method", ["Local (Frame-based Model)", "Deepware API"])

# Upload video
uploaded_file = st.file_uploader("Upload a video file (MP4, MOV)", type=["mp4", "mov", "avi"])

def dummy_fake_detector(frame: np.ndarray) -> float:
    """
    Dummy detector that returns a fake probability score.
    Replace with actual model inference.
    """
    return np.random.rand()

def call_deepware_api(video_path: str) -> dict:
    """
    Send video to Deepware AI API and get detection result.
    """
    api_url = "https://api.deepware.ai/v1/scan"
    api_key = "YOUR_DEEPWARE_API_KEY"  # Replace with your actual key

    with open(video_path, 'rb') as f:
        files = {'file': (os.path.basename(video_path), f)}
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(api_url, files=files, headers=headers)

    return response.json()

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(uploaded_file)

    if api_option == "Local (Frame-based Model)":
        st.info("Running frame-by-frame local DeepFake detection...")
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)
        fake_scores = []

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            if i % 15 == 0:  # Sample every 15th frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                score = dummy_fake_detector(frame_rgb)
                fake_scores.append(score)
            progress.progress(i / frame_count)

        avg_score = np.mean(fake_scores)
        st.success(f"📊 Average Fake Score: {avg_score:.2f}")
        if avg_score > 0.5:
            st.error("⚠️ The video is likely DeepFake!")
        else:
            st.success("✅ The video is likely genuine.")

    elif api_option == "Deepware API":
        st.info("Uploading to Deepware API...")
        result = call_deepware_api(video_path)

        if "result" in result:
            st.json(result)
            if result["result"]["label"] == "FAKE":
                st.error("⚠️ DeepFake Detected!")
            else:
                st.success("✅ No DeepFake Detected.")
        else:
            st.error("❌ Failed to get a valid response from the API.")

    os.remove(video_path)
