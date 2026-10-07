from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"


def connect() -> duckdb.DuckDBPyConnection:
    assert DB_PATH.exists()
    return duckdb.connect(str(DB_PATH), read_only=True)


def test_stage5_tables_exist() -> None:
    expected = {
        "mart_stage5_station_network",
        "mart_stage5_route_opportunities",
        "mart_stage5_city_market_opportunities",
        "mart_stage5_partner_route_opportunities",
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


def test_route_metrics_are_valid() -> None:
    with connect() as con:
        rows = con.execute(
            """
            SELECT
                distance_km,
                search_to_book_conversion,
                no_result_rate
            FROM mart_stage5_route_opportunities
            """
        ).fetchall()

    assert rows
    for distance, conversion, no_result in rows:
        assert distance > 0
        assert 0 <= conversion <= 1
        assert 0 <= no_result <= 1


def test_network_ranks_are_unique() -> None:
    with connect() as con:
        station_rows, station_ranks = con.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT station_opportunity_rank)
            FROM mart_stage5_station_network
            """
        ).fetchone()

        route_rows, route_ranks = con.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT opportunity_rank)
            FROM mart_stage5_route_opportunities
            """
        ).fetchone()

    assert station_rows == station_ranks
    assert route_rows == route_ranks


def test_stage5_artifacts_exist() -> None:
    assert (ROOT / "reports" / "generated" / "stage5" / "route_opportunity_map.html").exists()

    assert (ROOT / "reports" / "STAGE5_NETWORK_FINDINGS.md").exists()

    assert (ROOT / "docs" / "STAGE5_NETWORK_METHOD.md").exists()

    assert (ROOT / "reports" / "generated" / "stage5" / "validation_report.json").exists()
