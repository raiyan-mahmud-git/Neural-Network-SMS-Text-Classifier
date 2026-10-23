"""Authenticated REST API for SMS spam classification."""

import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from math import ceil
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, field_validator

import config
from predict import get_inference_model, predict_message

logger = logging.getLogger(__name__)

_RATE_LIMIT_LOCK = threading.Lock()
_RATE_WINDOW_STARTED = 0.0
_RATE_WINDOW_COUNT = 0


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Validate deployment configuration and warm the model before serving."""
    global _RATE_WINDOW_STARTED, _RATE_WINDOW_COUNT

    api_key = os.getenv("API_KEY", "")
    if len(api_key) < 32:
        raise RuntimeError("API_KEY must be configured with at least 32 characters")

    try:
        rate_limit = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "120"))
    except ValueError as exc:
        raise RuntimeError("API_RATE_LIMIT_PER_MINUTE must be a positive integer") from exc
    if rate_limit < 1:
        raise RuntimeError("API_RATE_LIMIT_PER_MINUTE must be a positive integer")

    application.state.api_key = api_key
    application.state.rate_limit_per_minute = rate_limit
    application.state.model_ready = False
    with _RATE_LIMIT_LOCK:
        _RATE_WINDOW_STARTED = time.monotonic()
        _RATE_WINDOW_COUNT = 0

    # Missing or invalid artifacts stop startup; serving never trains or downloads data.
    get_inference_model()
    application.state.model_ready = True
    yield
    application.state.model_ready = False


app = FastAPI(
    title="SMS Spam Detection Service",
    description="Authenticated real-time SMS spam scoring service.",
    version="1.0.0",
    lifespan=lifespan,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class SMSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The SMS text message to evaluate",
        examples=["Congratulations! You won a £1000 cash prize. Call 08001234 now!"],
    )
    threshold: float = Field(
        default=config.DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Decision threshold for spam classification",
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain non-whitespace text")
        return value


class SMSResponse(BaseModel):
    spam_probability: float
    label: str
    is_spam: bool
    threshold_applied: float


def require_api_key(
    request: Request,
    supplied_key: Annotated[str | None, Depends(api_key_header)],
) -> None:
    """Authenticate requests and apply a per-process request limit."""
    expected_key = request.app.state.api_key
    if supplied_key is None or not hmac.compare_digest(
        supplied_key.encode("utf-8"), expected_key.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    global _RATE_WINDOW_STARTED, _RATE_WINDOW_COUNT
    now = time.monotonic()
    with _RATE_LIMIT_LOCK:
        elapsed = now - _RATE_WINDOW_STARTED
        if elapsed >= 60:
            _RATE_WINDOW_STARTED = now
            _RATE_WINDOW_COUNT = 0
            elapsed = 0
        if _RATE_WINDOW_COUNT >= request.app.state.rate_limit_per_minute:
            retry_after = max(1, ceil(60 - elapsed))
            raise HTTPException(
                status_code=429,
                detail="Request rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        _RATE_WINDOW_COUNT += 1


@app.get("/health", tags=["Monitoring"])
def health_check(request: Request) -> dict[str, str | bool]:
    """Unauthenticated liveness/readiness probe for the container platform."""
    ready = bool(request.app.state.model_ready)
    return {
        "status": "healthy" if ready else "starting",
        "service": "sms-spam-classifier",
        "model_ready": ready,
    }


@app.post(
    "/predict",
    response_model=SMSResponse,
    tags=["Inference"],
    dependencies=[Depends(require_api_key)],
)
def predict_sms(request: SMSRequest) -> SMSResponse:
    """Classify an SMS message and return its spam probability."""
    try:
        prob, label = predict_message(request.message, threshold=request.threshold)
        return SMSResponse(
            spam_probability=round(prob, 4),
            label=label,
            is_spam=(label == "spam"),
            threshold_applied=request.threshold,
        )
    except Exception:
        logger.exception("SMS prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, workers=1)
