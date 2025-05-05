import streamlit as st
import cv2
import tempfile
import numpy as np
import requests
import os

# Page configuration
st.set_page_config(page_title="DeepFake Detector", layout="centered")

# Custom CSS with updated background and styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Open+Sans&display=swap');

        body {
            background-image: url('https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?auto=format&fit=crop&w=1950&q=80');
            background-size: cover;
            background-attachment: fixed;
        }

        .stApp {
            background-color: rgba(0, 0, 0, 0.65);
            padding: 2rem;
            border-radius: 12px;
        }

        h1 {
            font-family: 'Orbitron', sans-serif;
            color: #FFFFFF;
            text-shadow: 1px 1px 4px black;
        }

        .animated-desc {
            font-family: 'Open Sans', sans-serif;
            color: white;
            font-size: 18px;
            animation: slideInLeft 2s ease-out forwards;
            opacity: 0;
            transform: translateX(-100%);
            margin-top: 1rem;
        }

        @keyframes slideInLeft {
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        h2, h3 {
            font-family: 'Open Sans', sans-serif;
            color: #00ffee;
        }

        .stSelectbox > div {
            font-weight: bold;
        }

        .stProgress > div > div > div > div {
            background-color: #00ffcc;
        }

        .css-1v0mbdj, .stMarkdown {
            font-family: 'Open Sans', sans-serif;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# New deepfake-relevant logo image (clickable)
st.markdown("""
<a href="https://www.intel.com/content/www/us/en/research/fakecatcher.html" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/10778/10778298.png" width="100">
</a>
""", unsafe_allow_html=True)

# Title
st.title("Real-Time DeepFake Detection in Social Media Videos")

# Animated description with slide-in
st.markdown("""
<div class='animated-desc'>
Analyze your video content for potential deepfake alterations using cutting-edge frame-by-frame AI detection.
</div>
""", unsafe_allow_html=True)

# Select detection method
api_option = st.selectbox("🔍 Choose Detection Method", ["Local (Frame-based Model)", "Deepware API"])

# Upload video
uploaded_file = st.file_uploader("📤 Upload a video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

# Dummy detector function
def dummy_fake_detector(frame: np.ndarray) -> float:
    return np.random.rand()

# Deepware API call
def call_deepware_api(video_path: str) -> dict:
    api_url = "https://api.deepware.ai/v1/scan"
    api_key = "YOUR_DEEPWARE_API_KEY"  # Replace this with your actual API key

    with open(video_path, 'rb') as f:
        files = {'file': (os.path.basename(video_path), f)}
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(api_url, files=files, headers=headers)

    return response.json()

# Main detection logic
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(uploaded_file)

    if api_option == "Local (Frame-based Model)":
        st.info("🧠 Running local DeepFake detection...")
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)
        fake_scores = []

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i % 15 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                score = dummy_fake_detector(frame_rgb)
                fake_scores.append(score)

            progress.progress(min((i + 1) / frame_count, 1.0))

        cap.release()
        avg_score = np.mean(fake_scores)
        st.success(f"📊 Average Fake Score: **{avg_score:.2f}**")

        if avg_score > 0.5:
            st.error("⚠️ The video is likely DeepFake!")
        else:
            st.success("✅ The video appears genuine.")

    elif api_option == "Deepware API":
        st.info("📡 Sending to Deepware API...")
        result = call_deepware_api(video_path)

        if "result" in result:
            st.json(result)
            if result["result"]["label"] == "FAKE":
                st.error("⚠️ DeepFake Detected!")
            else:
                st.success("✅ No DeepFake Detected.")
        else:
            st.error("❌ API response error.")

    os.remove(video_path)
