from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"


def connect() -> duckdb.DuckDBPyConnection:
    assert DB_PATH.exists(), "Run scripts/build_stage2_warehouse.py first."
    return duckdb.connect(str(DB_PATH), read_only=True)


def test_expected_stage2_tables_exist() -> None:
    expected = {
        "fct_search_journey",
        "fct_session_journey",
        "mart_daily_product_metrics",
        "mart_weekly_product_metrics",
        "mart_partner_activation",
        "mart_partner_performance",
        "mart_partner_weekly",
        "mart_partner_retention",
        "mart_experiment_summary",
        "mart_experiment_effects",
        "mart_route_performance",
        "mart_kpi_summary",
    }
    with connect() as con:
        actual = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
    assert expected.issubset(actual)


def test_fact_grains_are_unique() -> None:
    with connect() as con:
        s_rows, s_unique = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT session_id) FROM fct_session_journey"
        ).fetchone()
        q_rows, q_unique = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT search_id) FROM fct_search_journey"
        ).fetchone()
    assert s_rows == s_unique
    assert q_rows == q_unique


def test_funnel_integrity_in_warehouse() -> None:
    with connect() as con:
        violations = con.execute(
            """
            SELECT COUNT(*)
            FROM fct_search_journey
            WHERE booked_flag = 1
              AND (selected_flag <> 1 OR no_result_flag = 1)
            """
        ).fetchone()[0]
    assert violations == 0


def test_experiment_arms_exist() -> None:
    with connect() as con:
        variants = {
            row[0]
            for row in con.execute(
                "SELECT experiment_variant FROM mart_experiment_summary"
            ).fetchall()
        }
    assert variants == {"control", "variant"}


def test_core_rates_are_probabilities() -> None:
    with connect() as con:
        rows = con.execute(
            """
            SELECT
                session_booking_conversion,
                search_to_book_conversion,
                no_result_rate,
                result_selection_rate,
                cancellation_rate,
                refund_rate
            FROM mart_daily_product_metrics
            """
        ).fetchall()
    assert rows
    for row in rows:
        for value in row:
            assert 0 <= value <= 1
