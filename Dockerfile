FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY individual_api ./individual_api
COPY historical_pipeline ./historical_pipeline
COPY MODEL_SPEC.json ./MODEL_SPEC.json

ARG PR31_MODEL_URL=https://github.com/nakasan3156-star/shogo-keirin-2shatan/releases/download/pr31-runtime-v1/pr31_frozen.joblib
ARG PR31_MODEL_SHA256=6aaaa9f1ce13db1e86b91b377997a8580be82948381d69097839a9b5673490e6
RUN mkdir -p /app/individual_api/models \
    && python -c "import hashlib, pathlib, urllib.request; p=pathlib.Path('/app/individual_api/models/pr31_frozen.joblib'); data=urllib.request.urlopen('${PR31_MODEL_URL}', timeout=60).read(); h=hashlib.sha256(data).hexdigest(); assert h == '${PR31_MODEL_SHA256}', f'PR31 model sha mismatch: {h}'; p.write_bytes(data); print('PR31 Frozen model verified', h)"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
