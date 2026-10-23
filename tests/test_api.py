"""Deployment contract tests for the authenticated HTTP API."""

import pytest
from fastapi.testclient import TestClient

from api import app

API_KEY = "test-key-that-is-long-enough-to-pass-startup-validation"


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "120")
    with TestClient(app) as client:
        yield client


def test_health_reports_model_ready_without_auth(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_ready"] is True


def test_predict_rejects_missing_or_invalid_key(api_client):
    payload = {"message": "Congratulations, you won a prize!"}
    assert api_client.post("/predict", json=payload).status_code == 401
    response = api_client.post(
        "/predict",
        json=payload,
        headers={"X-API-Key": "incorrect-key"},
    )
    assert response.status_code == 401


def test_predict_accepts_key_and_returns_valid_result(api_client):
    response = api_client.post(
        "/predict",
        json={"message": "sale today! to stop texts call 98912460324"},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "spam"
    assert result["is_spam"] is True
    assert 0.0 <= result["spam_probability"] <= 1.0
    assert result["threshold_applied"] == 0.95


def test_predict_hides_inference_exception_details(api_client, monkeypatch):
    def fail_prediction(*args, **kwargs):
        raise RuntimeError("internal file path")

    monkeypatch.setattr("api.predict_message", fail_prediction)
    response = api_client.post(
        "/predict",
        json={"message": "hello"},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed"
    assert "internal file path" not in response.text


def test_predict_validates_message_and_threshold(api_client):
    headers = {"X-API-Key": API_KEY}
    assert api_client.post("/predict", json={"message": "  "}, headers=headers).status_code == 422
    assert api_client.post(
        "/predict",
        json={"message": "hello", "threshold": 1.5},
        headers=headers,
    ).status_code == 422


def test_predict_enforces_rate_limit(monkeypatch):
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1")
    with TestClient(app) as client:
        headers = {"X-API-Key": API_KEY}
        payload = {"message": "hello", "threshold": 0.5}
        assert client.post("/predict", json=payload, headers=headers).status_code == 200
        limited = client.post("/predict", json=payload, headers=headers)
        assert limited.status_code == 429
        assert limited.headers["retry-after"]


def test_startup_requires_a_strong_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"), TestClient(app):
        pass
