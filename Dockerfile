FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    RAILNEXUS_ENABLE_DOCS=true

WORKDIR /app

COPY requirements-api.txt /app/requirements-api.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements-api.txt

RUN groupadd --system railnexus \
    && useradd --system --gid railnexus --home /app railnexus

COPY src /app/src
COPY app /app/app
COPY artifacts/models /app/artifacts/models
COPY data/processed/railnexus.duckdb /app/data/processed/railnexus.duckdb

RUN chown -R railnexus:railnexus /app

USER railnexus

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4)"

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
