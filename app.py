from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
import librosa
import joblib

app = Flask(__name__)

# Load Video Model
VIDEO_MODEL_PATH = "deepfake_xception_model.h5"
if not os.path.exists(VIDEO_MODEL_PATH):
    raise FileNotFoundError(f"Video model file not found: {VIDEO_MODEL_PATH}")

video_model = tf.keras.models.load_model(VIDEO_MODEL_PATH)

# Load Audio Model
AUDIO_MODEL_PATH = "trained_model.pkl"
SCALER_PATH = "scaler.pkl"
if not os.path.exists(AUDIO_MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Audio model or scaler file not found!")

audio_model = joblib.load(AUDIO_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------------------------
# Video Processing Functions
# ------------------------------------
def extract_frames(video_path, frame_interval=10, target_size=(299, 299)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frame_count += 1

    cap.release()
    
    if not frames:
        return np.array([])
    
    frames = np.array(frames, dtype=np.float32) / 255.0
    return frames

def detect_deepfake_video(video_path):
    frames = extract_frames(video_path)
    if frames.shape[0] == 0:
        return "Error: No valid frames extracted from the video."

    predictions = video_model.predict(frames)
    fake_count = np.sum(predictions > 0.5)
    real_count = len(predictions) - fake_count

    return "Fake Video" if fake_count > real_count else "Real Video"

# ------------------------------------
# Audio Processing Functions
# ------------------------------------
def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

def detect_deepfake_audio(audio_path):
    features = extract_mfcc_features(audio_path)
    if features is None:
        return "Error: Could not extract features from audio."

    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = audio_model.predict(features_scaled)

    return "Fake Audio" if prediction[0] == 1 else "Real Audio"

# ------------------------------------
# Flask Routes
# ------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_page")
def upload_page():
    return render_template("upload.html")

@app.route("/upload_video_audio", methods=["POST"])
def upload_video_audio():
    response = {}

    # Handle Video
    if "video" in request.files:
        video = request.files["video"]
        if video.filename != "":
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
            video.save(video_path)
            response["video_result"] = detect_deepfake_video(video_path)

    # Handle Audio
    if "audio" in request.files:
        audio = request.files["audio"]
        if audio.filename != "":
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio.filename)
            audio.save(audio_path)
            response["audio_result"] = detect_deepfake_audio(audio_path)

    return jsonify(response)

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
