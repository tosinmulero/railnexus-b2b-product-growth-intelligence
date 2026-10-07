from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(
    os.getenv(
        "RAILNEXUS_DB_PATH",
        str(ROOT / "data" / "processed" / "railnexus.duckdb"),
    )
)
MODEL_PATH = Path(
    os.getenv(
        "RAILNEXUS_MODEL_PATH",
        str(ROOT / "artifacts" / "models" / "stage6_booking_champion.joblib"),
    )
)
METADATA_PATH = Path(
    os.getenv(
        "RAILNEXUS_MODEL_METADATA_PATH",
        str(ROOT / "artifacts" / "models" / "stage6_model_metadata.json"),
    )
)
ENABLE_DOCS = os.getenv("RAILNEXUS_ENABLE_DOCS", "true").lower() == "true"

LOGGER = logging.getLogger("railnexus.api")
logging.basicConfig(
    level=os.getenv("RAILNEXUS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

RUNTIME: dict[str, Any] = {
    "model": None,
    "metadata": None,
}


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_type: str = Field(min_length=1, max_length=80)
    acquisition_channel: str = Field(min_length=1, max_length=80)
    traveller_type: str = Field(min_length=1, max_length=80)
    partner_type: str = Field(min_length=1, max_length=80)
    partner_country: str = Field(min_length=1, max_length=80)
    integration_type: str = Field(min_length=1, max_length=80)
    contract_tier: str = Field(min_length=1, max_length=80)
    journey_type: str = Field(min_length=1, max_length=80)
    origin_country: str = Field(min_length=1, max_length=80)
    destination_country: str = Field(min_length=1, max_length=80)
    search_dow: str = Field(min_length=1, max_length=20)

    is_logged_in: bool
    passengers: int = Field(ge=1, le=20)
    days_before_travel: int = Field(ge=0, le=730)
    result_count: int = Field(ge=0, le=500)
    no_result_flag: bool
    search_latency_ms: float = Field(ge=0, le=120000)
    international_flag: bool
    search_hour: int = Field(ge=0, le=23)


class PredictionResponse(BaseModel):
    model: str
    booking_probability: float
    locked_threshold: float
    predicted_booking: bool
    threshold_source: str
    data_classification: str
    prediction_point: str


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise RuntimeError(f"Model metadata not found: {METADATA_PATH}")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def open_db() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise RuntimeError(f"DuckDB database not found: {DB_PATH}")
    return duckdb.connect(str(DB_PATH), read_only=True)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(
        frame.to_json(
            orient="records",
            date_format="iso",
        )
    )


def scalar_row(frame: pd.DataFrame) -> dict[str, Any]:
    output = records(frame)
    if not output:
        raise HTTPException(status_code=404, detail="No data available.")
    return output[0]


def readiness() -> tuple[bool, dict[str, Any]]:
    checks = {
        "database_exists": DB_PATH.exists(),
        "model_exists": MODEL_PATH.exists(),
        "metadata_exists": METADATA_PATH.exists(),
        "model_loaded": RUNTIME["model"] is not None,
        "metadata_loaded": RUNTIME["metadata"] is not None,
    }
    return all(checks.values()), checks


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    RUNTIME["metadata"] = load_metadata()
    RUNTIME["model"] = joblib.load(MODEL_PATH)

    ready, checks = readiness()
    if not ready:
        raise RuntimeError(f"API readiness checks failed: {checks}")

    LOGGER.info(
        "RailNexus API ready model=%s database=%s",
        RUNTIME["metadata"].get("champion_model", "unknown"),
        DB_PATH,
    )
    yield
    RUNTIME["model"] = None
    RUNTIME["metadata"] = None


app = FastAPI(
    title="RailNexus B2B Product Intelligence API",
    version="1.0.0",
    description=(
        "Production-style API over synthetic RailNexus product analytics, "
        "experimentation, network intelligence and booking propensity."
    ),
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    lifespan=lifespan,
)

allowed_origins = [
    item.strip() for item in os.getenv("RAILNEXUS_CORS_ORIGINS", "").split(",") if item.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception(
            "Unhandled API error request_id=%s path=%s",
            request_id,
            request.url.path,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{elapsed_ms:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"

    LOGGER.info(
        "request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "RailNexus B2B Product Intelligence API",
        "version": "1.0.0",
        "data_classification": "synthetic portfolio data",
        "docs": "/docs" if ENABLE_DOCS else "disabled",
    }


@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready() -> dict[str, Any]:
    ready, checks = readiness()
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks},
        )
    return {"status": "ready", "checks": checks}


@app.get("/api/v1/overview")
def overview() -> dict[str, Any]:
    with open_db() as con:
        frame = con.execute(
            """
            SELECT
                COUNT(*) AS sessions,
                SUM(search_count) AS searches,
                SUM(booking_count) AS bookings,
                AVG(booked_flag) AS session_booking_conversion,
                SUM(successful_booking_value) AS successful_booking_value,
                SUM(platform_revenue) AS platform_revenue,
                COUNT(DISTINCT partner_id) AS active_partners
            FROM fct_session_journey
            """
        ).fetch_df()
    return scalar_row(frame)


@app.get("/api/v1/partners/opportunities")
def partner_opportunities(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    with open_db() as con:
        frame = con.execute(
            """
            SELECT
                priority_rank,
                partner_id,
                partner_type,
                country,
                integration_type,
                contract_tier,
                sessions,
                searches,
                session_booking_conversion,
                no_result_rate,
                conversion_change_pp,
                successful_value_growth_rate,
                opportunity_type,
                opportunity_score
            FROM mart_stage3_partner_opportunities
            ORDER BY priority_rank
            LIMIT ?
            """,
            [limit],
        ).fetch_df()
    return records(frame)


@app.get("/api/v1/routes/opportunities")
def route_opportunities(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    with open_db() as con:
        frame = con.execute(
            """
            SELECT
                opportunity_rank,
                origin_city,
                destination_city,
                origin_country,
                destination_country,
                searches,
                bookings,
                search_to_book_conversion,
                no_result_rate,
                distance_km,
                opportunity_type,
                underserved_opportunity_score
            FROM mart_stage5_route_opportunities
            ORDER BY opportunity_rank
            LIMIT ?
            """,
            [limit],
        ).fetch_df()
    return records(frame)


@app.get("/api/v1/experiment/smart-alternatives")
def experiment_summary() -> dict[str, Any]:
    with open_db() as con:
        primary = scalar_row(
            con.execute(
                """
                SELECT *
                FROM mart_stage4_primary_effect
                """
            ).fetch_df()
        )
        srm = scalar_row(
            con.execute(
                """
                SELECT *
                FROM mart_stage4_srm
                """
            ).fetch_df()
        )
        guardrails = records(
            con.execute(
                """
                SELECT *
                FROM mart_stage4_guardrails
                ORDER BY metric
                """
            ).fetch_df()
        )

    return {
        "experiment": "smart_alternatives_v1",
        "experimental_unit": "session",
        "primary_effect": primary,
        "sample_ratio_mismatch": srm,
        "guardrails": guardrails,
        "data_classification": "synthetic portfolio data",
    }


@app.get("/api/v1/anomalies")
def anomalies(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    with open_db() as con:
        frame = con.execute(
            """
            SELECT *
            FROM mart_stage7_daily_anomalies
            ORDER BY ABS(robust_z) DESC
            LIMIT ?
            """,
            [limit],
        ).fetch_df()
    return records(frame)


@app.get("/api/v1/model")
def model_metadata() -> dict[str, Any]:
    metadata = RUNTIME["metadata"]
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata unavailable.")

    return {
        "champion_model": metadata["champion_model"],
        "selection_metric": metadata["selection_metric"],
        "threshold_source": metadata["threshold_source"],
        "locked_threshold": float(metadata["locked_threshold"]),
        "prediction_point": metadata["prediction_point"],
        "features": metadata["features"],
        "excluded_leakage_features": metadata["excluded_leakage_features"],
        "data_classification": metadata["data_classification"],
    }


@app.post(
    "/api/v1/predict/booking",
    response_model=PredictionResponse,
)
def predict_booking(payload: PredictionRequest) -> PredictionResponse:
    model = RUNTIME["model"]
    metadata = RUNTIME["metadata"]

    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    row = {
        "device_type": payload.device_type,
        "acquisition_channel": payload.acquisition_channel,
        "traveller_type": payload.traveller_type,
        "partner_type": payload.partner_type,
        "partner_country": payload.partner_country,
        "integration_type": payload.integration_type,
        "contract_tier": payload.contract_tier,
        "journey_type": payload.journey_type,
        "origin_country": payload.origin_country,
        "destination_country": payload.destination_country,
        "search_dow": payload.search_dow,
        "is_logged_in": int(payload.is_logged_in),
        "passengers": payload.passengers,
        "days_before_travel": payload.days_before_travel,
        "result_count": payload.result_count,
        "no_result_flag": int(payload.no_result_flag),
        "search_latency_ms": payload.search_latency_ms,
        "international_flag": int(payload.international_flag),
        "search_hour": payload.search_hour,
    }

    features = list(metadata["features"])
    frame = pd.DataFrame([row], columns=features)
    probability = float(model.predict_proba(frame)[0, 1])
    threshold = float(metadata["locked_threshold"])

    return PredictionResponse(
        model=str(metadata["champion_model"]),
        booking_probability=probability,
        locked_threshold=threshold,
        predicted_booking=probability >= threshold,
        threshold_source=str(metadata["threshold_source"]),
        data_classification=str(metadata["data_classification"]),
        prediction_point=str(metadata["prediction_point"]),
    )
