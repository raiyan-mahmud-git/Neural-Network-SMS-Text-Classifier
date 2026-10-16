"""
Production REST API for SMS Spam Classification.
Built with FastAPI and Pydantic.

WHY THIS MATTERS:
In production microservice architectures, ML models are packaged behind RESTful
or gRPC APIs with strict request/response data contracts, schema validation,
and health check endpoints for Kubernetes/container orchestration.
"""


import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from predict import get_inference_model, predict_message

app = FastAPI(
    title="SMS Spam Detection Service",
    description="Microservice providing real-time SMS spam scoring powered by deep learning.",
    version="1.0.0"
)


class SMSRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="The SMS text message to evaluate",
                         examples=["Congratulations! You won a £1000 cash prize. Call 08001234 now!"])
    threshold: float = Field(default=config.DEFAULT_THRESHOLD, ge=0.0, le=1.0,
                             description="Decision threshold for spam classification")


class SMSResponse(BaseModel):
    message: str
    spam_probability: float
    label: str
    is_spam: bool
    threshold_applied: float


@app.on_event("startup")
def startup_event():
    """Pre-warm model into memory on server boot to avoid cold-start latency."""
    get_inference_model()


@app.get("/health", tags=["Monitoring"])
def health_check() -> dict[str, str]:
    """Liveness & readiness probe for container orchestrators."""
    return {"status": "healthy", "service": "sms-spam-classifier"}


@app.post("/predict", response_model=SMSResponse, tags=["Inference"])
def predict_sms(request: SMSRequest) -> SMSResponse:
    """Classifies an SMS message and returns probability, label, and metadata."""
    try:
        prob, label = predict_message(request.message, threshold=request.threshold)
        return SMSResponse(
            message=request.message,
            spam_probability=round(prob, 4),
            label=label,
            is_spam=(label == "spam"),
            threshold_applied=request.threshold
        )
    except Exception as e: # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference error: {e!s}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
