import streamlit as st
import os
import zipfile
import tempfile
import shutil
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="DeepFake Trainer", layout="centered")
st.title("🎭 DeepFake Detection: Upload Dataset & Train Model")

# --- Step 1: Upload dataset ZIP ---
uploaded_zip = st.file_uploader("📦 Upload ZIP file containing /real and /fake image folders", type=["zip"])

def extract_dataset(uploaded_zip):
    if uploaded_zip is not None:
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir
    return None

# --- Step 2: Load and preprocess images ---
def load_images_and_labels(dataset_dir, img_size=(64, 64)):
    data = []
    labels = []
    label_map = {"real": 0, "fake": 1}

    for label in ["real", "fake"]:
        folder = os.path.join(dataset_dir, label)
        if not os.path.exists(folder):
            st.error(f"❌ Folder '{label}' not found in uploaded ZIP.")
            return None, None
        for img_file in os.listdir(folder):
            try:
                img_path = os.path.join(folder, img_file)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                arr = img_to_array(img) / 255.0
                data.append(arr)
                labels.append(label_map[label])
            except:
                continue  # skip corrupt images

    X = np.array(data)
    y = to_categorical(np.array(labels), num_classes=2)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Train the model ---
def train_model(X_train, X_test, y_train, y_test):
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

# --- Step 4: Upload a video ---
uploaded_video = st.file_uploader("📤 Upload a video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

# --- Step 5: Run if dataset uploaded ---
if uploaded_zip:
    st.success("✅ Dataset ZIP uploaded!")
    dataset_path = extract_dataset(uploaded_zip)

    with st.spinner("📚 Loading data..."):
        X_train, X_test, y_train, y_test = load_images_and_labels(dataset_path)

    if X_train is not None:
        with st.spinner("🧠 Training model..."):
            model = train_model(X_train, X_test, y_train, y_test)
        st.success("✅ Model trained!")

        # --- Predict function ---
        def predict_frame(frame: np.ndarray) -> float:
            img = Image.fromarray(frame).resize((64, 64))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr)[0]
            return pred[1]  # 'fake' score

        # --- Analyze uploaded video ---
        if uploaded_video:
            st.video(uploaded_video)
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name

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
