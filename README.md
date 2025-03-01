# Deepfake_Detection_Audio_-_Video


# Overview

This project is a Deepfake Detection System that analyzes both video and audio files to determine if they have been manipulated using deepfake techniques. The system leverages deep learning for video detection and machine learning for audio detection, providing a comprehensive analysis for potential deepfake content.

# Features

Web-based Interface: Upload and analyze video and audio files separately.

Deep Learning for Video Analysis: Uses an Xception model to detect deepfake videos.

Machine Learning for Audio Analysis: Uses an SVM model to detect deepfake audio.

Flask Backend: Handles file uploads and processes video/audio deepfake detection.

JSON API Responses: Results are returned to the frontend for display.

# System Architecture

The system consists of the following components:

**Frontend (Web UI)**

Users upload video/audio files.

Separate buttons for processing video and audio.

Displays results in real time.

**Backend (Flask API)**

Handles video and audio file uploads.

Routes for processing video and audio separately.

Sends detection results back to the frontend.

**Video Deepfake Detection**

Extracts frames from the uploaded video.

Uses a pretrained Xception-based model (deepfake_xception_model.h5).

Outputs "Fake Video" or "Real Video".

**Audio Deepfake Detection**

Extracts MFCC features from audio.

Uses an SVM model (trained_model.pkl) for classification.

Outputs "Fake Audio" or "Real Audio".

**Installation Prerequisites**

Ensure you have the following installed:

Python 3.x

Flask

TensorFlow

OpenCV

Librosa

NumPy

Joblib

Setup

**Clone the repository:**

git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection

**Install required dependencies:**

pip install -r requirements.txt

**Run the Application**

**Start the Flask server:**

python app.py

**Access the web application at**:

http://127.0.0.1:5000/

Usage

Upload a video and/or audio file.

Click the "Process Video" or "Process Audio" button.

View the detection results.

**File Structure**

📂 deepfake-detection
│── app.py               # Flask application (backend)
│── static/              # Static files (CSS, JavaScript, assets)
│── templates/           # HTML templates (Frontend UI)
│── models/              # Pretrained models
│   ├── deepfake_xception_model.h5  # Video deepfake detection model
│   ├── trained_model.pkl           # Audio deepfake detection model
│   ├── scaler.pkl                   # Audio feature scaler
│── requirements.txt     # Dependencies
│── README.md            # Project documentation

**Contributing
**
Contributions are welcome! To contribute:

Fork the repository.

Create a feature branch (git checkout -b feature-name).

Commit your changes (git commit -m 'Add feature').

Push to your branch (git push origin feature-name).

Open a Pull Request.

**License
**
This project is licensed under the MIT License. See LICENSE for more details.

**Contact**

For inquiries, contact: devangbissa51@gmail.com
                        golegaonkarpurva177@gmail.com

**Acknowledgments**

TensorFlow & Keras for deep learning.

OpenCV for video processing.

Librosa for audio analysis.

Flask for backend development.
