import sys
import os
import joblib
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure model exists before running API tests
from src.train import train_and_save_model

MODEL_PATH = "models/model.pkl"

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model(output_path=MODEL_PATH)
    from app.main import app
    return TestClient(app)

def test_root_endpoint(setup_model):
    client = setup_model
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_health_endpoint(setup_model):
    client = setup_model
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_predict_endpoint_valid_payload(setup_model):
    client = setup_model
    payload = {
        "score_10th": 85.0,
        "score_12th": 88.5,
        "education_level": "Bachelor",
        "current_job": "Software Engineer",
        "years_of_experience": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_readiness_score" in data
    assert 0.0 <= data["predicted_readiness_score"] <= 100.0
    assert "readiness_tier" in data

def test_predict_endpoint_invalid_payload(setup_model):
    client = setup_model
    payload = {
        "score_10th": 150.0, # Invalid score > 100
        "score_12th": 88.5,
        "education_level": "Bachelor",
        "current_job": "Software Engineer",
        "years_of_experience": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
