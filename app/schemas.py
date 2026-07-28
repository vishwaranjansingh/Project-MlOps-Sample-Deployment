from pydantic import BaseModel, Field
from typing import Optional

class PredictionInput(BaseModel):
    score_10th: float = Field(..., ge=0.0, le=100.0, description="10th grade percentage or score (0-100)")
    score_12th: float = Field(..., ge=0.0, le=100.0, description="12th grade percentage or score (0-100)")
    education_level: str = Field(..., description="Highest education level e.g. High School, Bachelor, Master, PhD")
    current_job: str = Field(..., description="Current job domain or title e.g. Software Engineer, Data Scientist, Manager, etc.")
    years_of_experience: float = Field(..., ge=0.0, le=50.0, description="Years of relevant professional experience")

    model_config = {
        "json_schema_extra": {
            "example": {
                "score_10th": 88.5,
                "score_12th": 92.0,
                "education_level": "Master",
                "current_job": "Data Scientist",
                "years_of_experience": 4.5
            }
        }
    }

class PredictionOutput(BaseModel):
    status: str
    predicted_readiness_score: float = Field(..., description="Predicted Aptitude / Mind Readiness score (0-100)")
    readiness_tier: str = Field(..., description="Category tier: High, Medium, or Emerging")
    details: dict
