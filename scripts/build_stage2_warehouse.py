from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
EXPORTS = ROOT / "reports" / "generated" / "stage2"
DB_PATH = PROCESSED / "railnexus.duckdb"
RAW_NAMES = [
    "partners",
    "stations",
    "sessions",
    "searches",
    "bookings",
    "partner_events",
]


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    n = pd.to_numeric(numerator, errors="coerce").astype(float)
    d = pd.to_numeric(denominator, errors="coerce").astype(float)
    values = np.divide(
        n.to_numpy(),
        d.to_numpy(),
        out=np.zeros(len(n), dtype=float),
        where=d.to_numpy() != 0,
    )
    return pd.Series(values, index=n.index)


def scalar_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def load_raw() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name in RAW_NAMES:
        path = RAW / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw parquet: {path}")
        frames[name] = pd.read_parquet(path)
    return frames


def validate_raw(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    partners = frames["partners"]
    sessions = frames["sessions"]
    searches = frames["searches"]
    bookings = frames["bookings"]

    checks = {
        "duplicate_partner_ids": int(partners["partner_id"].duplicated().sum()),
        "duplicate_session_ids": int(sessions["session_id"].duplicated().sum()),
        "duplicate_search_ids": int(searches["search_id"].duplicated().sum()),
        "duplicate_booking_ids": int(bookings["booking_id"].duplicated().sum()),
        "orphan_session_partner": int((~sessions["partner_id"].isin(partners["partner_id"])).sum()),
        "orphan_search_session": int((~searches["session_id"].isin(sessions["session_id"])).sum()),
        "orphan_booking_search": int((~bookings["search_id"].isin(searches["search_id"])).sum()),
        "zero_result_selected": int(
            ((searches["no_result_flag"] == 1) & (searches["selected_flag"] == 1)).sum()
        ),
    }

    timeline = sessions[["partner_id", "session_ts"]].merge(
        partners[["partner_id", "launch_date"]],
        on="partner_id",
        how="left",
        validate="many_to_one",
    )
    timeline["session_ts"] = pd.to_datetime(timeline["session_ts"])
    timeline["launch_date"] = pd.to_datetime(timeline["launch_date"])
    checks["sessions_before_partner_launch"] = int(
        (timeline["session_ts"] < timeline["launch_date"]).sum()
    )

    booked_searches = searches.loc[
        searches["search_id"].isin(bookings["search_id"]),
        ["selected_flag", "no_result_flag"],
    ]
    checks["booking_without_selection"] = int((booked_searches["selected_flag"] != 1).sum())
    checks["booking_from_zero_result"] = int((booked_searches["no_result_flag"] == 1).sum())

    failures = {name: value for name, value in checks.items() if value != 0}
    if failures:
        raise RuntimeError(f"Raw data integrity failed: {failures}")
    return checks


def add_successful_value(bookings: pd.DataFrame) -> pd.DataFrame:
    result = bookings.copy()
    result["successful_booking_value"] = np.where(
        (result["cancelled_flag"] == 0) & (result["refund_flag"] == 0),
        result["booking_value"],
        0.0,
    )
    return result


def build_search_fact(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    searches = frames["searches"].copy()
    sessions = frames["sessions"][
        [
            "session_id",
            "partner_id",
            "device_type",
            "acquisition_channel",
            "is_logged_in",
            "traveller_type",
        ]
    ].copy()
    bookings = add_successful_value(frames["bookings"])

    booking_agg = bookings.groupby("search_id", as_index=False).agg(
        booking_count=("booking_id", "count"),
        booking_value=("booking_value", "sum"),
        successful_booking_value=("successful_booking_value", "sum"),
        platform_revenue=("platform_revenue", "sum"),
        cancellations=("cancelled_flag", "sum"),
        refunds=("refund_flag", "sum"),
    )

    fact = searches.merge(
        sessions,
        on=["session_id", "partner_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        booking_agg,
        on="search_id",
        how="left",
        validate="one_to_one",
    )

    fill_cols = [
        "booking_count",
        "booking_value",
        "successful_booking_value",
        "platform_revenue",
        "cancellations",
        "refunds",
    ]
    fact[fill_cols] = fact[fill_cols].fillna(0)
    fact["booked_flag"] = (fact["booking_count"] > 0).astype(int)
    fact["search_ts"] = pd.to_datetime(fact["search_ts"])
    fact["search_date"] = fact["search_ts"].dt.normalize()
    fact["search_week"] = fact["search_ts"].dt.to_period("W-SUN").dt.start_time
    fact["search_month"] = fact["search_ts"].dt.to_period("M").dt.start_time
    return fact


def build_session_fact(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sessions = frames["sessions"].copy()
    searches = frames["searches"].copy()
    bookings = add_successful_value(frames["bookings"])

    search_agg = searches.groupby("session_id", as_index=False).agg(
        search_count=("search_id", "count"),
        zero_result_searches=("no_result_flag", "sum"),
        selected_searches=("selected_flag", "sum"),
        alternatives_shown_count=("alternatives_shown", "sum"),
        total_search_latency_ms=("search_latency_ms", "sum"),
        avg_search_latency_ms=("search_latency_ms", "mean"),
    )
    booking_agg = bookings.groupby("session_id", as_index=False).agg(
        booking_count=("booking_id", "count"),
        booking_value=("booking_value", "sum"),
        successful_booking_value=("successful_booking_value", "sum"),
        platform_revenue=("platform_revenue", "sum"),
        cancellations=("cancelled_flag", "sum"),
        refunds=("refund_flag", "sum"),
    )

    fact = sessions.merge(
        search_agg,
        on="session_id",
        how="left",
        validate="one_to_one",
    ).merge(
        booking_agg,
        on="session_id",
        how="left",
        validate="one_to_one",
    )

    numeric = [
        "search_count",
        "zero_result_searches",
        "selected_searches",
        "alternatives_shown_count",
        "total_search_latency_ms",
        "avg_search_latency_ms",
        "booking_count",
        "booking_value",
        "successful_booking_value",
        "platform_revenue",
        "cancellations",
        "refunds",
    ]
    fact[numeric] = fact[numeric].fillna(0)
    fact["booked_flag"] = (fact["booking_count"] > 0).astype(int)
    fact["session_ts"] = pd.to_datetime(fact["session_ts"])
    fact["session_date"] = fact["session_ts"].dt.normalize()
    fact["session_week"] = fact["session_ts"].dt.to_period("W-SUN").dt.start_time
    fact["session_month"] = fact["session_ts"].dt.to_period("M").dt.start_time
    return fact


def metric_rollup(session_fact: pd.DataFrame, grain: str) -> pd.DataFrame:
    grouped = session_fact.groupby(grain, sort=True)
    out = grouped.agg(
        sessions=("session_id", "count"),
        active_partners=("partner_id", "nunique"),
        searches=("search_count", "sum"),
        bookings=("booking_count", "sum"),
        booking_sessions=("booked_flag", "sum"),
        zero_result_searches=("zero_result_searches", "sum"),
        selected_searches=("selected_searches", "sum"),
        alternatives_shown=("alternatives_shown_count", "sum"),
        total_search_latency_ms=("total_search_latency_ms", "sum"),
        booking_value=("booking_value", "sum"),
        successful_booking_value=("successful_booking_value", "sum"),
        platform_revenue=("platform_revenue", "sum"),
        cancellations=("cancellations", "sum"),
        refunds=("refunds", "sum"),
    ).reset_index()

    out["session_booking_conversion"] = safe_div(out["booking_sessions"], out["sessions"])
    out["search_to_book_conversion"] = safe_div(out["bookings"], out["searches"])
    out["no_result_rate"] = safe_div(out["zero_result_searches"], out["searches"])
    out["result_selection_rate"] = safe_div(out["selected_searches"], out["searches"])
    out["alternatives_exposure_rate"] = safe_div(out["alternatives_shown"], out["searches"])
    out["avg_search_latency_ms"] = safe_div(out["total_search_latency_ms"], out["searches"])
    out["cancellation_rate"] = safe_div(out["cancellations"], out["bookings"])
    out["refund_rate"] = safe_div(out["refunds"], out["bookings"])
    out["successful_booking_value_per_active_partner"] = safe_div(
        out["successful_booking_value"], out["active_partners"]
    )
    return out


def build_partner_activation(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    partners = frames["partners"].copy()
    events = frames["partner_events"].copy()
    events["event_ts"] = pd.to_datetime(events["event_ts"])

    pivot = events.pivot_table(
        index="partner_id",
        columns="event_type",
        values="event_ts",
        aggfunc="min",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "api_key_created": "api_key_created_ts",
            "integration_test": "integration_test_ts",
            "webhook_configured": "webhook_configured_ts",
            "first_search": "first_search_ts",
            "first_booking": "first_booking_ts",
        }
    )

    out = partners.merge(pivot, on="partner_id", how="left", validate="one_to_one")
    out["launch_date"] = pd.to_datetime(out["launch_date"])
    for col in [
        "api_key_created_ts",
        "integration_test_ts",
        "webhook_configured_ts",
        "first_search_ts",
        "first_booking_ts",
    ]:
        if col not in out.columns:
            out[col] = pd.NaT
        out[col] = pd.to_datetime(out[col])

    out["integration_completed_flag"] = out["integration_test_ts"].notna().astype(int)
    out["first_search_flag"] = out["first_search_ts"].notna().astype(int)
    out["first_booking_flag"] = out["first_booking_ts"].notna().astype(int)
    out["days_to_first_booking"] = (
        out["first_booking_ts"].dt.normalize() - out["launch_date"].dt.normalize()
    ).dt.days

    observation_end = pd.Timestamp("2026-06-30 23:59:59")
    out["activation_eligible_30d_flag"] = (
        out["launch_date"] <= observation_end - pd.Timedelta(days=30)
    ).astype(int)
    out["activated_30d_flag"] = (
        out["activation_eligible_30d_flag"].eq(1)
        & out["days_to_first_booking"].between(0, 30, inclusive="both")
    ).astype(int)
    return out


def build_partner_performance(partners: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    agg = fact.groupby("partner_id", as_index=False).agg(
        sessions=("session_id", "count"),
        searches=("search_count", "sum"),
        bookings=("booking_count", "sum"),
        booking_sessions=("booked_flag", "sum"),
        zero_result_searches=("zero_result_searches", "sum"),
        alternatives_shown=("alternatives_shown_count", "sum"),
        total_search_latency_ms=("total_search_latency_ms", "sum"),
        booking_value=("booking_value", "sum"),
        successful_booking_value=("successful_booking_value", "sum"),
        platform_revenue=("platform_revenue", "sum"),
        cancellations=("cancellations", "sum"),
        refunds=("refunds", "sum"),
    )
    out = partners[
        ["partner_id", "partner_type", "country", "integration_type", "contract_tier"]
    ].merge(agg, on="partner_id", how="left", validate="one_to_one")
    numeric = [c for c in agg.columns if c != "partner_id"]
    out[numeric] = out[numeric].fillna(0)
    out["session_booking_conversion"] = safe_div(out["booking_sessions"], out["sessions"])
    out["no_result_rate"] = safe_div(out["zero_result_searches"], out["searches"])
    out["alternatives_exposure_rate"] = safe_div(out["alternatives_shown"], out["searches"])
    out["avg_search_latency_ms"] = safe_div(out["total_search_latency_ms"], out["searches"])
    return out


def build_partner_weekly(fact: pd.DataFrame) -> pd.DataFrame:
    out = fact.groupby(["partner_id", "session_week"], as_index=False).agg(
        sessions=("session_id", "count"),
        searches=("search_count", "sum"),
        bookings=("booking_count", "sum"),
        booking_sessions=("booked_flag", "sum"),
        successful_booking_value=("successful_booking_value", "sum"),
        platform_revenue=("platform_revenue", "sum"),
    )
    out["session_booking_conversion"] = safe_div(out["booking_sessions"], out["sessions"])
    return out


def build_retention(fact: pd.DataFrame) -> pd.DataFrame:
    activity = fact[["partner_id", "session_month"]].drop_duplicates()
    first = (
        activity.groupby("partner_id", as_index=False)["session_month"]
        .min()
        .rename(columns={"session_month": "cohort_month"})
    )
    cohort = activity.merge(first, on="partner_id", validate="many_to_one")
    cohort["month_index"] = (
        (cohort["session_month"].dt.year - cohort["cohort_month"].dt.year) * 12
        + cohort["session_month"].dt.month
        - cohort["cohort_month"].dt.month
    )
    sizes = (
        first.groupby("cohort_month", as_index=False)["partner_id"]
        .nunique()
        .rename(columns={"partner_id": "cohort_size"})
    )
    active = (
        cohort.groupby(["cohort_month", "month_index"], as_index=False)["partner_id"]
        .nunique()
        .rename(columns={"partner_id": "active_partners"})
    )
    out = active.merge(sizes, on="cohort_month", validate="many_to_one")
    out["retention_rate"] = safe_div(out["active_partners"], out["cohort_size"])
    return out.sort_values(["cohort_month", "month_index"]).reset_index(drop=True)


def build_experiment(fact: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = fact.loc[fact["experiment_variant"].isin(["control", "variant"])].copy()
    if exp.empty:
        raise RuntimeError("Experiment window has no eligible sessions")

    summary = exp.groupby("experiment_variant", as_index=False).agg(
        sessions=("session_id", "count"),
        booking_sessions=("booked_flag", "sum"),
        searches=("search_count", "sum"),
        bookings=("booking_count", "sum"),
        zero_result_searches=("zero_result_searches", "sum"),
        alternatives_shown=("alternatives_shown_count", "sum"),
        total_search_latency_ms=("total_search_latency_ms", "sum"),
        cancellations=("cancellations", "sum"),
        refunds=("refunds", "sum"),
        booking_value=("booking_value", "sum"),
        successful_booking_value=("successful_booking_value", "sum"),
        platform_revenue=("platform_revenue", "sum"),
    )
    summary["session_booking_conversion"] = safe_div(
        summary["booking_sessions"], summary["sessions"]
    )
    summary["no_result_rate"] = safe_div(summary["zero_result_searches"], summary["searches"])
    summary["alternatives_exposure_rate"] = safe_div(
        summary["alternatives_shown"], summary["searches"]
    )
    summary["avg_search_latency_ms"] = safe_div(
        summary["total_search_latency_ms"], summary["searches"]
    )
    summary["cancellation_rate"] = safe_div(summary["cancellations"], summary["bookings"])
    summary["refund_rate"] = safe_div(summary["refunds"], summary["bookings"])

    by = summary.set_index("experiment_variant")
    if not {"control", "variant"}.issubset(by.index):
        raise RuntimeError("Both experiment arms are required")
    c, v = by.loc["control"], by.loc["variant"]
    effects = pd.DataFrame(
        [
            {
                "control_conversion": c["session_booking_conversion"],
                "variant_conversion": v["session_booking_conversion"],
                "absolute_conversion_uplift": v["session_booking_conversion"]
                - c["session_booking_conversion"],
                "relative_conversion_uplift": scalar_div(
                    v["session_booking_conversion"] - c["session_booking_conversion"],
                    c["session_booking_conversion"],
                ),
                "control_no_result_rate": c["no_result_rate"],
                "variant_no_result_rate": v["no_result_rate"],
                "no_result_rate_delta": v["no_result_rate"] - c["no_result_rate"],
                "control_latency_ms": c["avg_search_latency_ms"],
                "variant_latency_ms": v["avg_search_latency_ms"],
                "latency_delta_ms": v["avg_search_latency_ms"] - c["avg_search_latency_ms"],
                "control_cancellation_rate": c["cancellation_rate"],
                "variant_cancellation_rate": v["cancellation_rate"],
                "control_refund_rate": c["refund_rate"],
                "variant_refund_rate": v["refund_rate"],
            }
        ]
    )
    return summary, effects


def build_routes(search_fact: pd.DataFrame) -> pd.DataFrame:
    out = search_fact.groupby(["origin_station_id", "destination_station_id"], as_index=False).agg(
        searches=("search_id", "count"),
        booked_searches=("booked_flag", "sum"),
        bookings=("booking_count", "sum"),
        zero_result_searches=("no_result_flag", "sum"),
        avg_search_latency_ms=("search_latency_ms", "mean"),
        booking_value=("booking_value", "sum"),
        successful_booking_value=("successful_booking_value", "sum"),
        platform_revenue=("platform_revenue", "sum"),
    )
    out["search_to_book_conversion"] = safe_div(out["booked_searches"], out["searches"])
    out["no_result_rate"] = safe_div(out["zero_result_searches"], out["searches"])
    return out


def build_kpis(fact: pd.DataFrame, activation: pd.DataFrame) -> pd.DataFrame:
    sessions = float(len(fact))
    active_partners = float(fact["partner_id"].nunique())
    searches = float(fact["search_count"].sum())
    bookings = float(fact["booking_count"].sum())
    booking_sessions = float(fact["booked_flag"].sum())
    zero_results = float(fact["zero_result_searches"].sum())
    selected = float(fact["selected_searches"].sum())
    successful_value = float(fact["successful_booking_value"].sum())
    booking_value = float(fact["booking_value"].sum())
    revenue = float(fact["platform_revenue"].sum())
    eligible = activation.loc[activation["activation_eligible_30d_flag"].eq(1)]
    activation_rate = float(eligible["activated_30d_flag"].mean()) if len(eligible) else 0.0

    rows = [
        ("sessions", sessions),
        ("active_partners", active_partners),
        ("searches", searches),
        ("bookings", bookings),
        ("session_booking_conversion", scalar_div(booking_sessions, sessions)),
        ("search_to_book_conversion", scalar_div(bookings, searches)),
        ("no_result_rate", scalar_div(zero_results, searches)),
        ("result_selection_rate", scalar_div(selected, searches)),
        ("booking_value", booking_value),
        ("successful_booking_value", successful_value),
        ("platform_revenue", revenue),
        (
            "successful_booking_value_per_active_partner",
            scalar_div(successful_value, active_partners),
        ),
        ("integration_completion_rate", float(activation["integration_completed_flag"].mean())),
        ("activation_30d_rate", activation_rate),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    temp_name = f"_tmp_{name}"
    con.register(temp_name, df)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "{temp_name}"')
    finally:
        con.unregister(temp_name)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    frames = load_raw()
    raw_checks = validate_raw(frames)

    search_fact = build_search_fact(frames)
    session_fact = build_session_fact(frames)
    activation = build_partner_activation(frames)
    partner_performance = build_partner_performance(frames["partners"], session_fact)
    partner_weekly = build_partner_weekly(session_fact)
    retention = build_retention(session_fact)
    daily = metric_rollup(session_fact, "session_date").rename(
        columns={"session_date": "metric_date"}
    )
    weekly = metric_rollup(session_fact, "session_week").rename(
        columns={"session_week": "metric_week"}
    )
    experiment_summary, experiment_effects = build_experiment(session_fact)
    routes = build_routes(search_fact)
    kpis = build_kpis(session_fact, activation)

    tables = {
        "stg_partners": frames["partners"],
        "stg_stations": frames["stations"],
        "stg_sessions": frames["sessions"],
        "stg_searches": frames["searches"],
        "stg_bookings": frames["bookings"],
        "stg_partner_events": frames["partner_events"],
        "fct_search_journey": search_fact,
        "fct_session_journey": session_fact,
        "mart_daily_product_metrics": daily,
        "mart_weekly_product_metrics": weekly,
        "mart_partner_activation": activation,
        "mart_partner_performance": partner_performance,
        "mart_partner_weekly": partner_weekly,
        "mart_partner_retention": retention,
        "mart_experiment_summary": experiment_summary,
        "mart_experiment_effects": experiment_effects,
        "mart_route_performance": routes,
        "mart_kpi_summary": kpis,
    }

    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    try:
        for name, df in tables.items():
            write_table(con, name, df)
    finally:
        con.close()

    export_names = [name for name in tables if name.startswith("mart_")]
    for name in export_names:
        tables[name].to_csv(EXPORTS / f"{name}.csv", index=False)

    validation = {
        "raw_checks": raw_checks,
        "table_rows": {name: int(len(df)) for name, df in tables.items()},
    }
    (EXPORTS / "validation_report.json").write_text(
        json.dumps(validation, indent=2, default=str), encoding="utf-8"
    )

    metric_dict = dict(zip(kpis["metric"], kpis["value"], strict=True))
    effects = experiment_effects.iloc[0].to_dict()
    summary = {
        "data_classification": "synthetic portfolio data",
        "sessions": int(metric_dict["sessions"]),
        "active_partners": int(metric_dict["active_partners"]),
        "searches": int(metric_dict["searches"]),
        "bookings": int(metric_dict["bookings"]),
        "session_booking_conversion": metric_dict["session_booking_conversion"],
        "search_to_book_conversion": metric_dict["search_to_book_conversion"],
        "no_result_rate": metric_dict["no_result_rate"],
        "activation_30d_rate": metric_dict["activation_30d_rate"],
        "successful_booking_value_per_active_partner": metric_dict[
            "successful_booking_value_per_active_partner"
        ],
        "experiment_descriptive_effects": effects,
    }
    (EXPORTS / "stage2_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print("RAILNEXUS - STAGE 2 PRODUCT ANALYTICS WAREHOUSE")
    print("=" * 76)
    print(f"Sessions                    : {summary['sessions']:,}")
    print(f"Active partners             : {summary['active_partners']:,}")
    print(f"Searches                    : {summary['searches']:,}")
    print(f"Bookings                    : {summary['bookings']:,}")
    print(f"Session booking conversion  : {summary['session_booking_conversion']:.2%}")
    print(f"Search-to-book conversion   : {summary['search_to_book_conversion']:.2%}")
    print(f"No-result rate              : {summary['no_result_rate']:.2%}")
    print(f"30-day activation rate      : {summary['activation_30d_rate']:.2%}")
    print(
        "Successful value / partner  : "
        f"£{summary['successful_booking_value_per_active_partner']:,.2f}"
    )
    print("\nSMART ALTERNATIVES EXPERIMENT - DESCRIPTIVE ONLY")
    print(f"Control conversion          : {effects['control_conversion']:.2%}")
    print(f"Variant conversion          : {effects['variant_conversion']:.2%}")
    print(f"Absolute uplift             : {effects['absolute_conversion_uplift']:.2%}")
    print(f"No-result delta             : {effects['no_result_rate_delta']:.2%}")
    print(f"Latency delta               : {effects['latency_delta_ms']:,.2f} ms")
    print(f"\nDuckDB warehouse            : {DB_PATH.relative_to(ROOT)}")
    print(f"Stage 2 exports             : {EXPORTS.relative_to(ROOT)}")
    print("STAGE 2                     : PASS")
    print("=" * 76)


if __name__ == "__main__":
    main()
