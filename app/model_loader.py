import os
import joblib

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at path: '{MODEL_PATH}'. Ensure train.py has been run.")
        _model_instance = joblib.load(MODEL_PATH)
    return _model_instance
