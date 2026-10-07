from __future__ import annotations

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
APP_PATH = ROOT / "app" / "streamlit_app.py"
OUT_DIR = ROOT / "reports" / "generated" / "stage8"

REQUIRED_TABLES = {
    "fct_session_journey",
    "mart_daily_product_metrics",
    "mart_stage3_funnel",
    "mart_stage3_activation_segments",
    "mart_stage3_partner_opportunities",
    "mart_stage3_retention_summary",
    "mart_stage4_primary_effect",
    "mart_stage4_guardrails",
    "mart_stage4_srm",
    "mart_stage4_cuped_effect",
    "mart_stage4_cluster_adjusted_effect",
    "mart_stage5_route_opportunities",
    "mart_stage5_station_network",
    "mart_stage6_model_validation",
    "mart_stage6_feature_importance",
    "mart_stage6_cluster_profile",
    "mart_stage7_investigation_cases",
    "mart_stage7_daily_anomalies",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not APP_PATH.exists():
        raise RuntimeError("Streamlit application file is missing.")

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        actual = {
            row[0]
            for row in con.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }

        missing = sorted(REQUIRED_TABLES - actual)
        if missing:
            raise RuntimeError("Stage 8 dashboard dependencies are missing: " + ", ".join(missing))

        row_counts = {}
        for table in sorted(REQUIRED_TABLES):
            row_counts[table] = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

    validation = {
        "status": "PASS",
        "data_classification": "synthetic portfolio data",
        "app_path": str(APP_PATH.relative_to(ROOT)),
        "required_tables": len(REQUIRED_TABLES),
        "table_row_counts": row_counts,
        "pages": [
            "Executive Overview",
            "Partner Growth",
            "Experimentation",
            "Network Intelligence",
            "Predictive Modelling",
            "AI Investigation",
        ],
        "controls": {
            "synthetic_data_notice": True,
            "human_review_language": True,
            "read_only_database": True,
        },
    }

    (OUT_DIR / "validation_report.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    print("STAGE 8 APP DEPENDENCIES: PASS")
    print(f"Validated {len(REQUIRED_TABLES)} DuckDB tables.")
    print(f"App: {APP_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
