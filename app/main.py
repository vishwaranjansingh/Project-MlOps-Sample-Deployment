import pandas as pd
from fastapi import FastAPI, HTTPException, status
from app.schemas import PredictionInput, PredictionOutput
from app.model_loader import get_model, MODEL_PATH

app = FastAPI(
    title="Human Mind & Career Aptitude Evaluation API",
    description="Microservice for predicting career aptitude and mind readiness score based on educational history and job status.",
    version="1.0.0"
)

def categorize_tier(score: float) -> str:
    if score >= 80.0:
        return "High Readiness & Alignment"
    elif score >= 60.0:
        return "Moderate Readiness & Growing Potential"
    else:
        return "Emerging Readiness / Foundational Skill Phase"

@app.get("/", tags=["Health & Info"])
def root():
    return {
        "service": "Human Mind & Career Aptitude ML Serving API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health & Info"])
def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    try:
        model = get_model()
        if model is not None:
            return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service Unhealthy: {str(e)}"
        )
    return {"status": "healthy", "model_loaded": False}

@app.get("/info", tags=["Health & Info"])
def model_info():
    return {
        "model_path": MODEL_PATH,
        "features": [
            "score_10th",
            "score_12th",
            "education_level",
            "current_job",
            "years_of_experience"
        ]
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(payload: PredictionInput):
    try:
        model = get_model()
        
        # Prepare input dataframe
        input_data = pd.DataFrame([{
            'score_10th': payload.score_10th,
            'score_12th': payload.score_12th,
            'education_level': payload.education_level,
            'current_job': payload.current_job,
            'years_of_experience': payload.years_of_experience
        }])
        
        prediction = model.predict(input_data)[0]
        score = round(float(prediction), 2)
        tier = categorize_tier(score)
        
        return PredictionOutput(
            status="success",
            predicted_readiness_score=score,
            readiness_tier=tier,
            details={
                "candidate_education": f"{payload.education_level} (10th: {payload.score_10th}%, 12th: {payload.score_12th}%)",
                "current_job": payload.current_job,
                "experience_years": payload.years_of_experience
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction Error: {str(e)}"
        )
