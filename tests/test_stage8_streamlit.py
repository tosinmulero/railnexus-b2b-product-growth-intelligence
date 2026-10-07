import ast
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
OUT_DIR = ROOT / "reports" / "generated" / "stage8"


def test_streamlit_app_parses() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    ast.parse(source)


def test_streamlit_pages_are_present() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    for page in [
        "Executive Overview",
        "Partner Growth",
        "Experimentation",
        "Network Intelligence",
        "Predictive Modelling",
        "AI Investigation",
    ]:
        assert page in source


def test_streamlit_uses_read_only_duckdb() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert "read_only=True" in source


def test_stage8_dependencies_exist() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = {
            row[0]
            for row in con.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }

    assert "mart_stage3_funnel" in tables
    assert "mart_stage4_primary_effect" in tables
    assert "mart_stage5_route_opportunities" in tables
    assert "mart_stage6_model_validation" in tables
    assert "mart_stage7_investigation_cases" in tables


def test_stage8_validation_report() -> None:
    path = OUT_DIR / "validation_report.json"
    assert path.exists()

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["controls"]["synthetic_data_notice"]
    assert report["controls"]["read_only_database"]
