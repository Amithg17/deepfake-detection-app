import streamlit as st
import os
import cv2
import numpy as np
import tempfile
import shutil
import zipfile
import urllib.request
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="DeepFake Trainer", layout="centered")
st.title("🎭 DeepFake Detection: Train & Test")

# Step 1: Download Sample Dataset
@st.cache_data
def download_and_extract_dataset():
    dataset_url = "https://github.com/chukshith/deepfake-demo-dataset/raw/main/deepfake-mini-dataset.zip"
    dataset_dir = "deepfake_dataset"

    if not os.path.exists(dataset_dir):
        st.info("📥 Downloading dataset...")
        zip_path = "deepfake_dataset.zip"
        urllib.request.urlretrieve(dataset_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        os.remove(zip_path)

    return dataset_dir

# Step 2: Load and preprocess images
def load_images_and_labels(dataset_dir, img_size=(64, 64)):
    data = []
    labels = []
    label_map = {"real": 0, "fake": 1}

    for label in ["real", "fake"]:
        folder = os.path.join(dataset_dir, label)
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = Image.open(img_path).convert('RGB').resize(img_size)
            arr = img_to_array(img) / 255.0
            data.append(arr)
            labels.append(label_map[label])

    X = np.array(data)
    y = to_categorical(np.array(labels), num_classes=2)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
@st.cache_resource
def train_model():
    dataset_dir = download_and_extract_dataset()
    X_train, X_test, y_train, y_test = load_images_and_labels(dataset_dir)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)
    return model

model = train_model()

# Step 4: Upload video
uploaded_file = st.file_uploader("📤 Upload a video file (MP4)", type=["mp4", "mov", "avi"])

def predict_frame(frame: np.ndarray) -> float:
    img = Image.fromarray(frame).resize((64, 64))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0]
    return pred[1]  # Probability of 'fake'

if uploaded_file:
    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(uploaded_file)

    st.info("🔍 Analyzing uploaded video...")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores = []
    progress = st.progress(0)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % 15 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            score = predict_frame(frame_rgb)
            scores.append(score)
        progress.progress(min((i + 1) / frame_count, 1.0))

    cap.release()
    os.remove(video_path)

    if scores:
        avg_score = np.mean(scores)
        st.success(f"📊 Average Fake Score: {avg_score:.2f}")
        if avg_score > 0.5:
            st.error("⚠️ Likely a DeepFake!")
        else:
            st.success("✅ Likely a Real video.")
    else:
        st.warning("⚠️ No frames could be processed.")
