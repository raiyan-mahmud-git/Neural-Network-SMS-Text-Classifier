FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    SMS_MODEL_PATH=/app/artifacts/models/best_model.keras

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

COPY --chown=app:app requirements.lock ./requirements.lock
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.lock

COPY --chown=app:app api.py config.py data.py models.py predict.py ./
COPY --chown=app:app artifacts/models/best_model.keras ./artifacts/models/best_model.keras
COPY --chown=app:app artifacts/model_manifest.json ./artifacts/model_manifest.json
RUN mkdir -p /app/data /app/artifacts/plots \
    && chown -R app:app /app/data /app/artifacts

USER app:app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import json, urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3))['model_ready']"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
