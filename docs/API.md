# RailNexus API

## Purpose

RailNexus exposes the validated synthetic B2B product-intelligence layers as a
production-style FastAPI service.

No proprietary Trainline, partner, traveller or commercial data is used.

## Run locally

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `/docs` for the generated OpenAPI interface.

## Core endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health/live` | Process liveness |
| GET | `/health/ready` | Database/model readiness |
| GET | `/api/v1/overview` | Product KPIs |
| GET | `/api/v1/partners/opportunities` | Ranked B2B partner opportunities |
| GET | `/api/v1/routes/opportunities` | Ranked rail-network opportunities |
| GET | `/api/v1/experiment/smart-alternatives` | Experiment effect and guardrails |
| GET | `/api/v1/anomalies` | Product anomaly queue |
| GET | `/api/v1/model` | Model governance metadata |
| POST | `/api/v1/predict/booking` | Booking propensity prediction |

## Prediction contract

The model prediction point is after first-search results are available and
before booking outcome.

The API only accepts the leakage-aware Stage 6 feature set. Extra fields are
rejected.

The decision threshold was selected on the validation period and locked before
the temporal test set was evaluated.

## Example prediction request

```json
{
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
  "is_logged_in": true,
  "passengers": 1,
  "days_before_travel": 14,
  "result_count": 8,
  "no_result_flag": false,
  "search_latency_ms": 420.0,
  "international_flag": false,
  "search_hour": 10
}
```
