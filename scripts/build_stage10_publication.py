from __future__ import annotations

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
REPORTS = ROOT / "reports" / "generated"
DOCS = ROOT / "docs"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Required publication artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> dict:
    frame = con.execute(sql).fetch_df()
    if frame.empty:
        raise RuntimeError("Publication query returned no rows.")
    return frame.iloc[0].to_dict()


def pct(value: float) -> str:
    return f"{float(value):.2%}"


def money(value: float) -> str:
    return f"£{float(value):,.0f}"


def number(value: float | int) -> str:
    return f"{int(value):,}"


def main() -> None:
    if not DB_PATH.exists():
        raise RuntimeError(f"DuckDB database not found: {DB_PATH}")

    stage6 = load_json(REPORTS / "stage6" / "champion_test_metrics.json")
    stage6_meta = load_json(ROOT / "artifacts" / "models" / "stage6_model_metadata.json")
    stage7 = load_json(REPORTS / "stage7" / "validation_report.json")
    stage9 = load_json(REPORTS / "stage9" / "validation_report.json")

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        overview = scalar(
            con,
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
            """,
        )

        experiment = scalar(
            con,
            """
            SELECT
                control_rate,
                variant_rate,
                absolute_effect,
                relative_effect,
                ci_low,
                ci_high,
                p_value
            FROM mart_stage4_primary_effect
            """,
        )

        srm = scalar(
            con,
            """
            SELECT p_value, srm_flag
            FROM mart_stage4_srm
            """,
        )

        network = scalar(
            con,
            """
            SELECT
                COUNT(*) AS routes,
                COUNT(DISTINCT origin_station_id)
                    + COUNT(DISTINCT destination_station_id) AS endpoint_count
            FROM mart_stage5_route_opportunities
            """,
        )

        station_count = scalar(
            con,
            """
            SELECT COUNT(*) AS stations
            FROM mart_stage5_station_network
            """,
        )["stations"]

        top_route = scalar(
            con,
            """
            SELECT
                origin_city,
                destination_city,
                searches,
                search_to_book_conversion,
                no_result_rate,
                opportunity_type
            FROM mart_stage5_route_opportunities
            ORDER BY opportunity_rank
            LIMIT 1
            """,
        )

        top_partner = scalar(
            con,
            """
            SELECT
                partner_id,
                opportunity_type,
                sessions,
                searches,
                session_booking_conversion,
                no_result_rate
            FROM mart_stage3_partner_opportunities
            ORDER BY priority_rank
            LIMIT 1
            """,
        )

        model_validation = con.execute(
            """
            SELECT model, pr_auc, roc_auc, log_loss, brier
            FROM mart_stage6_model_validation
            ORDER BY pr_auc DESC
            """
        ).fetch_df()

        cluster_profile = con.execute(
            """
            SELECT *
            FROM mart_stage6_cluster_profile
            ORDER BY cluster_id
            """
        ).fetch_df()

        anomaly_count = scalar(
            con,
            """
            SELECT COUNT(*) AS cases
            FROM mart_stage7_investigation_cases
            """,
        )["cases"]

    champion = str(stage6_meta["champion_model"])
    champion_validation = model_validation.loc[model_validation["model"] == champion]
    validation_pr_auc = (
        float(champion_validation.iloc[0]["pr_auc"])
        if not champion_validation.empty
        else float("nan")
    )

    architecture = """# RailNexus Architecture

```mermaid
flowchart LR
    A[Synthetic B2B Rail Generator] --> B[Parquet Data Layer]
    B --> C[DuckDB Analytics Warehouse]
    C --> D[Product KPI & Funnel Marts]
    C --> E[Experimentation Layer]
    C --> F[Graph & Geospatial Layer]
    C --> G[Predictive Modelling]
    C --> H[Anomaly & Investigation Layer]

    D --> I[Streamlit Product App]
    E --> I
    F --> I
    G --> I
    H --> I

    G --> J[Persisted Champion Model]
    C --> K[FastAPI]
    J --> K

    K --> L[Docker]
    L --> M[CI/CD Quality Gate]

    N[Human Review / Responsible AI] --> H
```

## Design principles

- **Synthetic by construction:** no proprietary Trainline, partner, traveller
  or commercial data is used.
- **Reproducible analytics:** deterministic generation, DuckDB marts and
  automated validation.
- **Methodological separation:** descriptive analytics, experimentation,
  predictive modelling and anomaly triage have distinct responsibilities.
- **Leakage-aware ML:** post-booking outcomes and treatment exposure are
  excluded from booking-propensity features.
- **Locked evaluation:** model selection and operating threshold come from the
  validation period before temporal test evaluation.
- **Human-in-the-loop AI:** anomaly packets separate observations from
  hypotheses and require human review.
- **Production-style serving:** FastAPI, Pydantic contracts, read-only DuckDB,
  model metadata, Docker and CI/CD.
"""
    (DOCS / "ARCHITECTURE.md").write_text(architecture, encoding="utf-8")

    recruiter = f"""# RailNexus — Recruiter Summary

## 60-second summary

RailNexus is an end-to-end **B2B rail product intelligence system** built with
synthetic data to demonstrate product analytics, experimentation, machine
learning, graph/geospatial analysis, responsible AI and production engineering.

The project analyses {number(overview["sessions"])} sessions,
{number(overview["searches"])} searches and {number(overview["bookings"])}
bookings across {number(overview["active_partners"])} active synthetic B2B
partners.

It includes:

- activation, funnel, retention and partner-growth analytics;
- session-level A/B testing with SRM checks, confidence intervals, CUPED-style
  adjustment, clustered inference, power/MDE and guardrails;
- route/station graph analysis with PageRank, betweenness and geospatial
  opportunity scoring;
- Logistic Regression, Probit, Random Forest and MLP candidate models;
- K-Means partner segmentation;
- validation-selected champion model with a validation-locked operating
  threshold;
- AI-assisted anomaly investigation with explicit human review;
- Streamlit product application;
- FastAPI model/analytics service;
- Docker and GitHub Actions CI.

## Current synthetic results

- Session booking conversion: **{pct(overview["session_booking_conversion"])}**
- Successful booking value: **{money(overview["successful_booking_value"])}**
- Experiment control conversion: **{pct(experiment["control_rate"])}**
- Experiment variant conversion: **{pct(experiment["variant_rate"])}**
- Experiment absolute effect: **{pct(experiment["absolute_effect"])}**
- Experiment p-value: **{float(experiment["p_value"]):.6f}**
- SRM p-value: **{float(srm["p_value"]):.6f}**
- Champion model: **{champion}**
- Validation PR-AUC: **{validation_pr_auc:.4f}**
- Locked test ROC-AUC: **{float(stage6["roc_auc"]):.4f}**
- Locked test PR-AUC: **{float(stage6["pr_auc"]):.4f}**
- Test top-decile lift: **{float(stage6["top_decile_lift"]):.3f}x**
- Partner clusters: **{len(cluster_profile)}**
- Network stations: **{number(station_count)}**
- Network routes: **{number(network["routes"])}**
- Investigation cases: **{number(anomaly_count)}**

All results above are **synthetic portfolio results**, not Trainline results.
"""
    (DOCS / "RECRUITER_SUMMARY.md").write_text(recruiter, encoding="utf-8")

    case_study = f"""# RailNexus Case Study

## Problem

A B2B rail platform needs a coherent way to understand partner activation,
traveller product friction, experiment impact, network opportunity and future
booking propensity while keeping analytics reproducible and decision-making
governed.

## Approach

RailNexus models the full analytical workflow:

1. deterministic synthetic B2B rail data generation;
2. DuckDB warehouse and product metric marts;
3. partner activation, funnel, retention and growth analysis;
4. rigorous session-level product experimentation;
5. graph/geospatial opportunity analysis;
6. predictive modelling and clustering;
7. AI-assisted anomaly investigation with human review;
8. Streamlit product application;
9. FastAPI, Docker and CI/CD production hardening.

## Product analytics

The synthetic system contains **{number(overview["sessions"])} sessions**,
**{number(overview["searches"])} searches** and
**{number(overview["bookings"])} bookings**. Overall session booking conversion
is **{pct(overview["session_booking_conversion"])}**.

The highest-ranked current partner opportunity is
**{top_partner["partner_id"]}**, classified as
**{top_partner["opportunity_type"]}**. The opportunity engine combines traffic,
commercial value, conversion friction, no-result exposure and recent movement;
it is a prioritisation heuristic rather than a causal score.

## Experimentation

Smart Journey Alternatives is evaluated with session-level intent-to-treat
analysis.

- Control conversion: **{pct(experiment["control_rate"])}**
- Variant conversion: **{pct(experiment["variant_rate"])}**
- Absolute effect: **{pct(experiment["absolute_effect"])}**
- 95% CI: **[{pct(experiment["ci_low"])}, {pct(experiment["ci_high"])}]**
- p-value: **{float(experiment["p_value"]):.6f}**
- SRM p-value: **{float(srm["p_value"]):.6f}**

The project also includes CUPED-style pre-period adjustment, partner-clustered
inference, power/MDE, guardrails and FDR-corrected exploratory heterogeneity.

## Network intelligence

The graph layer analyses **{number(station_count)} stations** and
**{number(network["routes"])} directed routes** using demand-weighted PageRank,
betweenness centrality and Haversine distance.

The highest-ranked route opportunity in the current synthetic run is
**{top_route["origin_city"]} → {top_route["destination_city"]}**, classified as
**{top_route["opportunity_type"]}**.

## Predictive modelling

Candidate models include Logistic Regression, Probit, Random Forest and MLP,
plus a dummy baseline.

The champion is selected using validation PR-AUC. Its operating threshold is
selected from validation predictions and locked before May–June temporal test
evaluation.

- Champion: **{champion}**
- Locked test ROC-AUC: **{float(stage6["roc_auc"]):.4f}**
- Locked test PR-AUC: **{float(stage6["pr_auc"]):.4f}**
- Locked test F1: **{float(stage6["f1"]):.4f}**
- Top-decile lift: **{float(stage6["top_decile_lift"]):.3f}x**

The feature contract excludes post-booking outcomes, revenue, experiment
assignment and Smart Alternatives exposure.

## Responsible AI

Daily metrics are monitored against rolling robust baselines. Structured
investigation packets contain observed evidence, segment/partner/route context,
hypotheses to test and recommended owners. Every case requires human review and
the workflow forbids unsupported causal claims.

Current investigation cases: **{number(stage7["cases_created"])}**.

## Production engineering

The validated application includes:

- Streamlit multi-page product application;
- FastAPI analytics and booking-propensity endpoints;
- liveness and readiness probes;
- Pydantic contracts;
- request IDs and security headers;
- read-only DuckDB at serving time;
- non-root Docker image and healthcheck;
- hardened Docker Compose settings;
- GitHub Actions CI for deterministic rebuild, lint, tests and Docker build.

Stage 9 production controls report: **{stage9["status"]}**.

## Data notice

Every partner, traveller, route, commercial, experiment and model value in this
repository is synthetic. The project does not claim actual Trainline product or
financial impact.
"""
    (DOCS / "CASE_STUDY.md").write_text(case_study, encoding="utf-8")

    readme = f"""# RailNexus

## B2B Rail Product Growth, Experimentation & Network Intelligence

RailNexus is an end-to-end **product data science and analytics platform** for
a fictional European B2B rail business. It combines product analytics,
experimentation, predictive modelling, partner segmentation, graph/geospatial
analysis, responsible AI, a Streamlit application and a production-style
FastAPI/Docker/CI layer.

> **Data notice:** all partner, traveller, commercial, experiment and network
> data in this repository is synthetic. No proprietary Trainline or customer
> data is used.

## What this project demonstrates

- B2B partner activation, retention, engagement and growth analytics
- search → results → selection → booking funnel measurement
- product metric trees and launch guardrails
- A/B testing with SRM, confidence intervals, CUPED-style adjustment,
  partner-clustered inference, power/MDE and multiple-testing control
- graph analysis using weighted PageRank and betweenness centrality
- geospatial route opportunity analysis using Haversine distance
- Logistic Regression, Probit, Random Forest and MLP modelling
- K-Means partner segmentation with silhouette-based model selection
- validation-based model selection and a **validation-locked threshold**
- AI-assisted anomaly triage with mandatory human review
- Streamlit product analytics application
- FastAPI model and analytics service
- Docker, healthchecks and GitHub Actions CI

## Current synthetic system

| Metric | Result |
| --- | ---: |
| Sessions | {number(overview["sessions"])} |
| Searches | {number(overview["searches"])} |
| Bookings | {number(overview["bookings"])} |
| Active B2B partners | {number(overview["active_partners"])} |
| Session booking conversion | {pct(overview["session_booking_conversion"])} |
| Successful booking value | {money(overview["successful_booking_value"])} |
| Network stations | {number(station_count)} |
| Directed network routes | {number(network["routes"])} |
| AI investigation cases | {number(anomaly_count)} |

## Product experiment

**Smart Journey Alternatives v1**

| Metric | Result |
| --- | ---: |
| Control conversion | {pct(experiment["control_rate"])} |
| Variant conversion | {pct(experiment["variant_rate"])} |
| Absolute effect | {pct(experiment["absolute_effect"])} |
| 95% CI | {pct(experiment["ci_low"])} to {pct(experiment["ci_high"])} |
| p-value | {float(experiment["p_value"]):.6f} |
| SRM p-value | {float(srm["p_value"]):.6f} |

The experimental unit is the **session**. The project also evaluates guardrails,
pre-period CUPED-style adjustment, partner-clustered uncertainty, statistical
power/MDE and exploratory heterogeneous effects with false-discovery-rate
control.

## Predictive modelling

Candidate models:

- Logistic Regression
- Probit
- Random Forest
- Multi-Layer Perceptron
- Dummy baseline

**Champion: {champion}**

| Locked test metric | Result |
| --- | ---: |
| ROC-AUC | {float(stage6["roc_auc"]):.4f} |
| PR-AUC | {float(stage6["pr_auc"]):.4f} |
| Log loss | {float(stage6["log_loss"]):.4f} |
| Brier score | {float(stage6["brier"]):.4f} |
| F1 | {float(stage6["f1"]):.4f} |
| Top-decile lift | {float(stage6["top_decile_lift"]):.3f}x |

The champion is selected on the validation window. Its operating threshold is
also selected on validation data and locked before temporal test evaluation.

## Network opportunity

The current highest-ranked synthetic route opportunity is:

**{top_route["origin_city"]} → {top_route["destination_city"]}**

Classification: **{top_route["opportunity_type"]}**

The route prioritisation layer combines demand, conversion gap, no-result
friction, endpoint network centrality and route distance. It is a prioritisation
heuristic, not causal inference.

## Architecture

```mermaid
flowchart LR
    A[Synthetic Data] --> B[Parquet]
    B --> C[DuckDB Warehouse]
    C --> D[Product Analytics]
    C --> E[Experimentation]
    C --> F[Graph + Geospatial]
    C --> G[Predictive ML]
    C --> H[AI Investigation]
    D --> I[Streamlit]
    E --> I
    F --> I
    G --> I
    H --> I
    G --> J[Model Artifact]
    C --> K[FastAPI]
    J --> K
    K --> L[Docker]
    L --> M[GitHub Actions CI]
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the detailed design.

## Technology

**Analytics & data:** Python, pandas, SQL, DuckDB, Parquet  
**Statistics:** SciPy, statsmodels, experiment design, confidence intervals,
power/MDE, CUPED-style adjustment  
**Machine learning:** scikit-learn, Logistic Regression, Random Forest, MLP,
K-Means, permutation importance  
**Network analysis:** NetworkX, PageRank, betweenness centrality  
**Visualisation/app:** Plotly, Streamlit  
**Serving/MLOps:** FastAPI, Pydantic, joblib, Docker, GitHub Actions, pytest,
Ruff

## Repository structure

```text
railnexus-b2b-product-growth-intelligence/
├── app/                    # Streamlit and FastAPI applications
├── config/                 # Project configuration
├── docs/                   # Architecture, experiment/model/API documentation
├── scripts/                # Reproducible pipeline stages
├── src/railnexus/          # Core synthetic-data package
├── tests/                  # Automated tests
├── reports/                # Executive findings and generated outputs
├── artifacts/              # Local model artifacts
├── data/                   # Local raw/processed synthetic data
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Reproduce the project

Create/activate the virtual environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-api.txt
```

Generate a deterministic smoke dataset:

```powershell
python scripts/generate_synthetic_data.py --sessions 10000
```

Build the analytical stages:

```powershell
python scripts/build_stage2_warehouse.py
python scripts/build_stage3_growth_analysis.py
python scripts/build_stage4_experiment.py
python scripts/build_stage5_network_intelligence.py
python scripts/build_stage6_predictive_modelling.py
python scripts/build_stage7_ai_investigation.py
python scripts/validate_stage8_app.py
python scripts/validate_stage9_production.py
```

Run validation:

```powershell
python -m ruff format --check src scripts tests app
python -m ruff check src scripts tests app
python -m pytest -q
```

## Run the product application

```powershell
python -m streamlit run app/streamlit_app.py
```

## Run the API

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `/docs` for the OpenAPI interface.

## Run with Docker

```powershell
docker compose up --build
```

## Key documentation

- [`docs/RECRUITER_SUMMARY.md`](docs/RECRUITER_SUMMARY.md)
- [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/EXPERIMENT_CARD.md`](docs/EXPERIMENT_CARD.md)
- [`docs/STAGE6_MODEL_CARD.md`](docs/STAGE6_MODEL_CARD.md)
- [`docs/AI_ASSISTED_INVESTIGATION.md`](docs/AI_ASSISTED_INVESTIGATION.md)
- [`docs/API.md`](docs/API.md)
- [`docs/PRODUCTION_HARDENING.md`](docs/PRODUCTION_HARDENING.md)

## Quality controls

The project includes deterministic data generation, grain/integrity checks,
Ruff formatting/linting, pytest, API contract tests, live container smoke tests,
Docker healthchecks and GitHub Actions CI.

## Portfolio positioning

RailNexus is designed to demonstrate the intersection of **Senior Data
Science, Product Analytics, Experimentation, B2B Growth Analytics, Machine
Learning and Analytics Engineering** in a rail-commerce context.

It is a portfolio project and does not claim employment at, internal knowledge
of, or access to proprietary data from Trainline.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    publication = {
        "status": "PASS",
        "data_classification": "synthetic portfolio data",
        "readme_generated": True,
        "architecture_generated": True,
        "case_study_generated": True,
        "recruiter_summary_generated": True,
        "champion_model": champion,
        "validation_threshold_source": stage6_meta["threshold_source"],
        "stage9_status": stage9["status"],
    }
    (REPORTS / "stage10").mkdir(parents=True, exist_ok=True)
    (REPORTS / "stage10" / "publication_manifest.json").write_text(
        json.dumps(publication, indent=2),
        encoding="utf-8",
    )

    print("STAGE 10 PUBLICATION CONTENT: PASS")
    print(f"Champion model: {champion}")
    print(f"README: {ROOT / 'README.md'}")
    print(f"Architecture: {DOCS / 'ARCHITECTURE.md'}")
    print(f"Case study: {DOCS / 'CASE_STUDY.md'}")
    print(f"Recruiter summary: {DOCS / 'RECRUITER_SUMMARY.md'}")


if __name__ == "__main__":
    main()
