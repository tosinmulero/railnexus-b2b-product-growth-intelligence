from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"


def connect() -> duckdb.DuckDBPyConnection:
    assert DB_PATH.exists()
    return duckdb.connect(str(DB_PATH), read_only=True)


def test_stage4_tables_exist() -> None:
    expected = {
        "mart_stage4_srm",
        "mart_stage4_balance",
        "mart_stage4_primary_effect",
        "mart_stage4_cuped_effect",
        "mart_stage4_cluster_adjusted_effect",
        "mart_stage4_guardrails",
        "mart_stage4_segment_effects",
        "mart_stage4_business_impact",
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


def test_primary_effect_is_valid() -> None:
    with connect() as con:
        row = con.execute(
            """
            SELECT
                control_rate,
                variant_rate,
                absolute_effect,
                ci_low,
                ci_high,
                p_value
            FROM mart_stage4_primary_effect
            """
        ).fetchone()

    control, variant, effect, low, high, p_value = row

    assert 0 <= control <= 1
    assert 0 <= variant <= 1
    assert low <= effect <= high
    assert 0 <= p_value <= 1


def test_no_sample_ratio_mismatch_fields_missing() -> None:
    with connect() as con:
        row = con.execute(
            """
            SELECT
                control_sessions,
                variant_sessions,
                variant_share,
                p_value
            FROM mart_stage4_srm
            """
        ).fetchone()

    control_n, variant_n, share, p_value = row

    assert control_n > 0
    assert variant_n > 0
    assert 0 < share < 1
    assert 0 <= p_value <= 1


def test_guardrail_p_values_are_valid() -> None:
    with connect() as con:
        rows = con.execute(
            """
            SELECT p_value
            FROM mart_stage4_guardrails
            """
        ).fetchall()

    assert rows
    for (p_value,) in rows:
        assert 0 <= p_value <= 1


def test_stage4_documents_exist() -> None:
    assert (ROOT / "reports" / "STAGE4_EXPERIMENT_FINDINGS.md").exists()

    assert (ROOT / "docs" / "EXPERIMENT_CARD.md").exists()

    assert (ROOT / "reports" / "generated" / "stage4" / "validation_report.json").exists()
