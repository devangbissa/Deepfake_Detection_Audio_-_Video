import pickle

AUDIO_MODEL_PATH = "svm_model.pkl"

try:
    with open(AUDIO_MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
except pickle.UnpicklingError:
    print("❌ ERROR: Invalid pickle file (corrupted or incorrect format).")
except FileNotFoundError:
    print("❌ ERROR: File not found!")
except Exception as e:
    print(f"❌ ERROR: {e}")

