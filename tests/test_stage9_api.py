from pathlib import Path

from fastapi.testclient import TestClient

from app.api import app

ROOT = Path(__file__).resolve().parents[1]


def test_api_health_and_readiness() -> None:
    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "alive"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_overview_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/overview")

    assert response.status_code == 200
    body = response.json()

    assert body["sessions"] > 0
    assert body["searches"] > 0
    assert body["active_partners"] > 0
    assert 0 <= body["session_booking_conversion"] <= 1


def test_ranked_opportunity_endpoints() -> None:
    with TestClient(app) as client:
        partners = client.get(
            "/api/v1/partners/opportunities",
            params={"limit": 5},
        )
        routes = client.get(
            "/api/v1/routes/opportunities",
            params={"limit": 5},
        )

    assert partners.status_code == 200
    assert routes.status_code == 200
    assert 1 <= len(partners.json()) <= 5
    assert 1 <= len(routes.json()) <= 5


def test_model_metadata_exposes_validation_threshold() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/model")

    assert response.status_code == 200
    body = response.json()

    assert body["threshold_source"] == "validation"
    assert 0 <= body["locked_threshold"] <= 1
    assert body["prediction_point"]


def test_booking_prediction_contract() -> None:
    payload = {
        "device_type": "Desktop",
        "acquisition_channel": "Direct",
        "traveller_type": "Leisure",
        "partner_type": "Online Travel Retailer",
        "partner_country": "United Kingdom",
        "integration_type": "API",
        "contract_tier": "Growth",
        "journey_type": "One Way",
        "origin_country": "United Kingdom",
        "destination_country": "United Kingdom",
        "search_dow": "Wednesday",
        "is_logged_in": True,
        "passengers": 1,
        "days_before_travel": 14,
        "result_count": 8,
        "no_result_flag": False,
        "search_latency_ms": 420.0,
        "international_flag": False,
        "search_hour": 10,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict/booking",
            json=payload,
        )

    assert response.status_code == 200
    body = response.json()

    assert 0 <= body["booking_probability"] <= 1
    assert 0 <= body["locked_threshold"] <= 1
    assert body["threshold_source"] == "validation"
    assert body["data_classification"] == "synthetic portfolio data"


def test_prediction_rejects_unknown_fields() -> None:
    payload = {
        "device_type": "Desktop",
        "acquisition_channel": "Direct",
        "traveller_type": "Leisure",
        "partner_type": "Online Travel Retailer",
        "partner_country": "United Kingdom",
        "integration_type": "API",
        "contract_tier": "Growth",
        "journey_type": "One Way",
        "origin_country": "United Kingdom",
        "destination_country": "United Kingdom",
        "search_dow": "Wednesday",
        "is_logged_in": True,
        "passengers": 1,
        "days_before_travel": 14,
        "result_count": 8,
        "no_result_flag": False,
        "search_latency_ms": 420.0,
        "international_flag": False,
        "search_hour": 10,
        "post_booking_revenue": 999999,
    }

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict/booking",
            json=payload,
        )

    assert response.status_code == 422


def test_security_headers_and_request_id() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/health/live",
            headers={"X-Request-ID": "stage9-test"},
        )

    assert response.headers["X-Request-ID"] == "stage9-test"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"
