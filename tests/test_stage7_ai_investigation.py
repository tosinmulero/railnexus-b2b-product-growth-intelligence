import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
OUT_DIR = ROOT / "reports" / "generated" / "stage7"


def connect() -> duckdb.DuckDBPyConnection:
    assert DB_PATH.exists()
    return duckdb.connect(str(DB_PATH), read_only=True)


def test_stage7_tables_exist() -> None:
    expected = {
        "mart_stage7_daily_anomalies",
        "mart_stage7_investigation_cases",
    }

    with connect() as con:
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

    assert expected.issubset(actual)


def test_human_review_gate_is_enforced() -> None:
    with connect() as con:
        rows = con.execute(
            """
            SELECT human_review_required
            FROM mart_stage7_investigation_cases
            """
        ).fetchall()

    for (required,) in rows:
        assert bool(required)


def test_context_packets_are_json_safe() -> None:
    path = OUT_DIR / "ai_context_packets.json"
    assert path.exists()

    packets = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(packets, list)

    for packet in packets:
        assert packet["human_review_required"] is True
        assert packet["data_classification"] == "synthetic portfolio data"


def test_stage7_validation_report_passes() -> None:
    path = OUT_DIR / "validation_report.json"
    assert path.exists()

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["checks"]["case_packet_count_matches"]
    assert report["checks"]["human_review_required"]


def test_stage7_documents_exist() -> None:
    expected = [
        OUT_DIR / "daily_metric_anomalies.csv",
        OUT_DIR / "investigation_case_register.csv",
        OUT_DIR / "ai_investigation_prompt.md",
        OUT_DIR / "ai_context_packets.json",
        ROOT / "docs" / "AI_ASSISTED_INVESTIGATION.md",
        ROOT / "reports" / "STAGE7_INVESTIGATION_FINDINGS.md",
    ]

    for path in expected:
        assert path.exists(), str(path)
