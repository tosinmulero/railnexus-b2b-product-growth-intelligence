from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"


def connect() -> duckdb.DuckDBPyConnection:
    assert DB_PATH.exists()
    return duckdb.connect(str(DB_PATH), read_only=True)


def test_stage3_tables_exist() -> None:
    expected = {
        "mart_stage3_funnel",
        "mart_stage3_activation_segments",
        "mart_stage3_product_segments",
        "mart_stage3_partner_growth",
        "mart_stage3_partner_opportunities",
        "mart_stage3_retention_summary",
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


def test_funnel_is_monotonic() -> None:
    with connect() as con:
        rows = con.execute(
            """
            SELECT stage_order, count
            FROM mart_stage3_funnel
            ORDER BY stage_order
            """
        ).fetchall()

    counts = [row[1] for row in rows]
    assert counts == sorted(counts, reverse=True)


def test_funnel_rates_are_valid() -> None:
    with connect() as con:
        rows = con.execute(
            """
            SELECT
                conversion_from_previous,
                conversion_from_search,
                dropoff_from_previous
            FROM mart_stage3_funnel
            """
        ).fetchall()

    for row in rows:
        for value in row:
            assert 0 <= value <= 1


def test_partner_opportunity_ranks_are_unique() -> None:
    with connect() as con:
        rows, unique_ranks = con.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT priority_rank)
            FROM mart_stage3_partner_opportunities
            """
        ).fetchone()

    assert rows == unique_ranks


def test_stage3_findings_exist() -> None:
    assert (ROOT / "reports" / "generated" / "stage3" / "stage3_findings.json").exists()

    assert (ROOT / "reports" / "STAGE3_EXECUTIVE_FINDINGS.md").exists()
