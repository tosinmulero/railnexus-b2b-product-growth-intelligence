from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
OUT_DIR = ROOT / "reports" / "generated" / "stage3"
FINDINGS_MD = ROOT / "reports" / "STAGE3_EXECUTIVE_FINDINGS.md"


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    n = pd.to_numeric(numerator, errors="coerce").astype(float)
    d = pd.to_numeric(denominator, errors="coerce").astype(float)
    return pd.Series(
        np.divide(
            n,
            d,
            out=np.zeros(len(n), dtype=float),
            where=d.to_numpy() != 0,
        ),
        index=n.index,
    )


def scalar_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    std = float(s.std(ddof=0))
    if std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - float(s.mean())) / std


def load_tables(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    names = [
        "stg_partners",
        "fct_session_journey",
        "fct_search_journey",
        "mart_partner_activation",
        "mart_partner_performance",
        "mart_partner_weekly",
        "mart_partner_retention",
    ]
    return {name: con.execute(f'SELECT * FROM "{name}"').fetch_df() for name in names}


def build_funnel(search_fact: pd.DataFrame) -> pd.DataFrame:
    searches = float(len(search_fact))
    with_results = float((search_fact["no_result_flag"] == 0).sum())
    selections = float(search_fact["selected_flag"].sum())
    booked_searches = float(search_fact["booked_flag"].sum())

    rows = [
        ("Searches", searches),
        ("Results returned", with_results),
        ("Journey selected", selections),
        ("Booked", booked_searches),
    ]
    funnel = pd.DataFrame(rows, columns=["stage", "count"])

    funnel["conversion_from_previous"] = [
        1.0,
        scalar_div(with_results, searches),
        scalar_div(selections, with_results),
        scalar_div(booked_searches, selections),
    ]
    funnel["conversion_from_search"] = [
        1.0,
        scalar_div(with_results, searches),
        scalar_div(selections, searches),
        scalar_div(booked_searches, searches),
    ]
    funnel["dropoff_from_previous"] = 1 - funnel["conversion_from_previous"]
    funnel.loc[0, "dropoff_from_previous"] = 0.0
    funnel["stage_order"] = [1, 2, 3, 4]
    return funnel


def build_activation_segments(
    partners: pd.DataFrame,
    activation: pd.DataFrame,
) -> pd.DataFrame:
    base = activation.merge(
        partners[
            [
                "partner_id",
                "partner_type",
                "country",
                "integration_type",
                "contract_tier",
            ]
        ],
        on=[
            "partner_id",
            "partner_type",
            "country",
            "integration_type",
            "contract_tier",
        ],
        how="left",
        validate="one_to_one",
    )

    outputs: list[pd.DataFrame] = []
    for dimension in [
        "partner_type",
        "country",
        "integration_type",
        "contract_tier",
    ]:
        eligible = base.loc[base["activation_eligible_30d_flag"] == 1].copy()
        grouped = eligible.groupby(dimension, as_index=False).agg(
            partners=("partner_id", "nunique"),
            integration_completion_rate=(
                "integration_completed_flag",
                "mean",
            ),
            first_search_rate=("first_search_flag", "mean"),
            first_booking_rate=("first_booking_flag", "mean"),
            activation_30d_rate=("activated_30d_flag", "mean"),
            median_days_to_first_booking=(
                "days_to_first_booking",
                "median",
            ),
        )
        grouped = grouped.rename(columns={dimension: "segment_value"})
        grouped.insert(0, "dimension", dimension)
        outputs.append(grouped)

    return pd.concat(outputs, ignore_index=True)


def build_product_segments(session_fact: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        ["device_type"],
        ["traveller_type"],
        ["acquisition_channel"],
        ["device_type", "traveller_type"],
    ]
    outputs: list[pd.DataFrame] = []

    for dims in dimensions:
        grouped = session_fact.groupby(dims, as_index=False).agg(
            sessions=("session_id", "count"),
            partners=("partner_id", "nunique"),
            searches=("search_count", "sum"),
            bookings=("booking_count", "sum"),
            booking_sessions=("booked_flag", "sum"),
            zero_result_searches=("zero_result_searches", "sum"),
            selected_searches=("selected_searches", "sum"),
            successful_booking_value=("successful_booking_value", "sum"),
            platform_revenue=("platform_revenue", "sum"),
        )

        grouped["session_booking_conversion"] = safe_div(
            grouped["booking_sessions"],
            grouped["sessions"],
        )
        grouped["search_to_book_conversion"] = safe_div(
            grouped["bookings"],
            grouped["searches"],
        )
        grouped["no_result_rate"] = safe_div(
            grouped["zero_result_searches"],
            grouped["searches"],
        )
        grouped["result_selection_rate"] = safe_div(
            grouped["selected_searches"],
            grouped["searches"],
        )

        grouped["dimension"] = " x ".join(dims)
        grouped["segment_value"] = grouped[dims].astype(str).agg(" | ".join, axis=1)

        keep = [
            "dimension",
            "segment_value",
            "sessions",
            "partners",
            "searches",
            "bookings",
            "session_booking_conversion",
            "search_to_book_conversion",
            "no_result_rate",
            "result_selection_rate",
            "successful_booking_value",
            "platform_revenue",
        ]
        outputs.append(grouped[keep])

    return pd.concat(outputs, ignore_index=True)


def build_partner_growth(
    partner_weekly: pd.DataFrame,
    partner_perf: pd.DataFrame,
) -> pd.DataFrame:
    weekly = partner_weekly.copy()
    weekly["session_week"] = pd.to_datetime(weekly["session_week"])

    max_week = weekly["session_week"].max()
    recent_start = max_week - pd.Timedelta(weeks=3)
    prior_end = recent_start - pd.Timedelta(days=1)
    prior_start = prior_end - pd.Timedelta(weeks=3)

    def window_rollup(
        frame: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        prefix: str,
    ) -> pd.DataFrame:
        subset = frame.loc[(frame["session_week"] >= start) & (frame["session_week"] <= end)]
        return subset.groupby("partner_id", as_index=False).agg(
            **{
                f"{prefix}_sessions": ("sessions", "sum"),
                f"{prefix}_searches": ("searches", "sum"),
                f"{prefix}_bookings": ("bookings", "sum"),
                f"{prefix}_booking_sessions": (
                    "booking_sessions",
                    "sum",
                ),
                f"{prefix}_successful_value": (
                    "successful_booking_value",
                    "sum",
                ),
                f"{prefix}_platform_revenue": (
                    "platform_revenue",
                    "sum",
                ),
            }
        )

    recent = window_rollup(
        weekly,
        recent_start,
        max_week,
        "recent_4w",
    )
    prior = window_rollup(
        weekly,
        prior_start,
        prior_end,
        "prior_4w",
    )

    growth = partner_perf.merge(
        recent,
        on="partner_id",
        how="left",
        validate="one_to_one",
    ).merge(
        prior,
        on="partner_id",
        how="left",
        validate="one_to_one",
    )

    window_cols = [
        c for c in growth.columns if c.startswith("recent_4w_") or c.startswith("prior_4w_")
    ]
    growth[window_cols] = growth[window_cols].fillna(0)

    growth["recent_4w_conversion"] = safe_div(
        growth["recent_4w_booking_sessions"],
        growth["recent_4w_sessions"],
    )
    growth["prior_4w_conversion"] = safe_div(
        growth["prior_4w_booking_sessions"],
        growth["prior_4w_sessions"],
    )
    growth["conversion_change_pp"] = growth["recent_4w_conversion"] - growth["prior_4w_conversion"]
    growth["successful_value_growth_rate"] = np.where(
        growth["prior_4w_successful_value"] > 0,
        (growth["recent_4w_successful_value"] / growth["prior_4w_successful_value"]) - 1,
        np.where(growth["recent_4w_successful_value"] > 0, 1.0, 0.0),
    )

    growth["recent_window_start"] = recent_start
    growth["recent_window_end"] = max_week
    growth["prior_window_start"] = prior_start
    growth["prior_window_end"] = prior_end

    return growth


def build_opportunities(partner_growth: pd.DataFrame) -> pd.DataFrame:
    df = partner_growth.copy()

    conversion_gap = (
        float(df["session_booking_conversion"].median()) - df["session_booking_conversion"]
    ).clip(lower=0)
    no_result_excess = (df["no_result_rate"] - float(df["no_result_rate"].median())).clip(lower=0)
    volume_signal = np.log1p(df["searches"])
    value_signal = np.log1p(df["successful_booking_value"])
    recent_decline = (-df["conversion_change_pp"]).clip(lower=0)

    df["opportunity_score"] = (
        0.30 * zscore(volume_signal)
        + 0.20 * zscore(value_signal)
        + 0.22 * zscore(conversion_gap)
        + 0.18 * zscore(no_result_excess)
        + 0.10 * zscore(recent_decline)
    )

    df["opportunity_type"] = np.select(
        [
            (
                (df["searches"] >= df["searches"].median())
                & (df["session_booking_conversion"] < df["session_booking_conversion"].median())
            ),
            df["no_result_rate"] > df["no_result_rate"].quantile(0.75),
            df["conversion_change_pp"] < 0,
            df["successful_value_growth_rate"] > 0.20,
        ],
        [
            "High-volume conversion opportunity",
            "Search-result availability opportunity",
            "Recent conversion deterioration",
            "Growth / expansion candidate",
        ],
        default="Monitor",
    )

    df["priority_rank"] = df["opportunity_score"].rank(method="first", ascending=False).astype(int)

    return df.sort_values(["priority_rank", "partner_id"]).reset_index(drop=True)


def build_retention_summary(retention: pd.DataFrame) -> pd.DataFrame:
    base = retention.copy()
    base["cohort_month"] = pd.to_datetime(base["cohort_month"])

    pivot = base.pivot_table(
        index="cohort_month",
        columns="month_index",
        values="retention_rate",
        aggfunc="first",
    )

    rows = []
    for cohort_month, row in pivot.iterrows():
        record = {
            "cohort_month": cohort_month,
            "month_0_retention": row.get(0, np.nan),
            "month_1_retention": row.get(1, np.nan),
            "month_2_retention": row.get(2, np.nan),
            "month_3_retention": row.get(3, np.nan),
        }
        rows.append(record)

    return pd.DataFrame(rows).sort_values("cohort_month").reset_index(drop=True)


def write_table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    df: pd.DataFrame,
) -> None:
    temp = f"_stage3_{name}"
    con.register(temp, df)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "{temp}"')
    finally:
        con.unregister(temp)


def generate_findings(
    funnel: pd.DataFrame,
    activation: pd.DataFrame,
    segments: pd.DataFrame,
    opportunities: pd.DataFrame,
    retention: pd.DataFrame,
) -> dict[str, object]:
    funnel_drop = (
        funnel.loc[funnel["stage_order"] > 1]
        .sort_values("dropoff_from_previous", ascending=False)
        .iloc[0]
    )

    eligible_activation = activation.loc[activation["dimension"] == "integration_type"].sort_values(
        "activation_30d_rate", ascending=True
    )

    weakest_activation = eligible_activation.iloc[0] if not eligible_activation.empty else None

    meaningful_segments = segments.loc[
        segments["sessions"] >= max(25, int(segments["sessions"].median()))
    ].copy()
    weakest_segment = (
        meaningful_segments.sort_values(
            "session_booking_conversion",
            ascending=True,
        ).iloc[0]
        if not meaningful_segments.empty
        else None
    )

    top_opportunity = opportunities.iloc[0]

    retention_with_m1 = retention.dropna(subset=["month_1_retention"])
    best_retention = (
        retention_with_m1.sort_values(
            "month_1_retention",
            ascending=False,
        ).iloc[0]
        if not retention_with_m1.empty
        else None
    )

    findings: dict[str, object] = {
        "data_classification": "synthetic portfolio data",
        "largest_funnel_drop": {
            "stage": str(funnel_drop["stage"]),
            "dropoff_rate": float(funnel_drop["dropoff_from_previous"]),
            "conversion_from_previous": float(funnel_drop["conversion_from_previous"]),
        },
        "top_partner_opportunity": {
            "partner_id": str(top_opportunity["partner_id"]),
            "opportunity_type": str(top_opportunity["opportunity_type"]),
            "priority_rank": int(top_opportunity["priority_rank"]),
            "sessions": int(top_opportunity["sessions"]),
            "searches": int(top_opportunity["searches"]),
            "session_booking_conversion": float(top_opportunity["session_booking_conversion"]),
            "no_result_rate": float(top_opportunity["no_result_rate"]),
        },
    }

    if weakest_activation is not None:
        findings["weakest_activation_segment"] = {
            "dimension": "integration_type",
            "segment": str(weakest_activation["segment_value"]),
            "activation_30d_rate": float(weakest_activation["activation_30d_rate"]),
            "partners": int(weakest_activation["partners"]),
        }

    if weakest_segment is not None:
        findings["weakest_product_segment"] = {
            "dimension": str(weakest_segment["dimension"]),
            "segment": str(weakest_segment["segment_value"]),
            "sessions": int(weakest_segment["sessions"]),
            "session_booking_conversion": float(weakest_segment["session_booking_conversion"]),
            "no_result_rate": float(weakest_segment["no_result_rate"]),
        }

    if best_retention is not None:
        findings["best_month1_retention_cohort"] = {
            "cohort_month": str(pd.Timestamp(best_retention["cohort_month"]).date()),
            "month_1_retention": float(best_retention["month_1_retention"]),
        }

    return findings


def write_findings_markdown(findings: dict[str, object]) -> None:
    largest = findings["largest_funnel_drop"]
    top = findings["top_partner_opportunity"]

    lines = [
        "# RailNexus — Stage 3 Executive Findings",
        "",
        "> Data notice: all partner, traveller, commercial and product values "
        "in this report are synthetic portfolio data.",
        "",
        "## Product Growth Summary",
        "",
        (
            f"- The largest funnel loss occurs at **{largest['stage']}**, "
            f"where the drop-off from the previous stage is "
            f"**{largest['dropoff_rate']:.2%}**."
        ),
        (
            f"- The highest-ranked partner opportunity is **{top['partner_id']}** "
            f"(**{top['opportunity_type']}**), based on a combination of "
            "traffic, value, conversion friction, no-result exposure and "
            "recent conversion movement."
        ),
        "",
    ]

    activation = findings.get("weakest_activation_segment")
    if activation:
        lines.extend(
            [
                "## Partner Activation",
                "",
                (
                    f"- The weakest integration-type segment for 30-day "
                    f"activation is **{activation['segment']}**, with an "
                    f"activation rate of **{activation['activation_30d_rate']:.2%}** "
                    f"across {activation['partners']} eligible partners."
                ),
                "",
            ]
        )

    product = findings.get("weakest_product_segment")
    if product:
        lines.extend(
            [
                "## Product Experience",
                "",
                (
                    f"- The weakest sufficiently sized product segment is "
                    f"**{product['segment']}** ({product['dimension']}), with "
                    f"session booking conversion of "
                    f"**{product['session_booking_conversion']:.2%}** and a "
                    f"no-result rate of **{product['no_result_rate']:.2%}**."
                ),
                "",
            ]
        )

    retention = findings.get("best_month1_retention_cohort")
    if retention:
        lines.extend(
            [
                "## Retention",
                "",
                (
                    f"- The strongest observable Month-1 partner retention "
                    f"cohort begins **{retention['cohort_month']}**, with "
                    f"Month-1 retention of "
                    f"**{retention['month_1_retention']:.2%}**."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Recommended Product Questions for Stage 4",
            "",
            "- Is the Smart Journey Alternatives uplift statistically reliable?",
            "- Which partner or traveller segments respond differently to the feature?",
            "- Are conversion gains achieved without materially worsening latency, "
            "cancellations or refunds?",
            "- What sample size and minimum detectable effect are required for "
            "future product launches?",
            "",
            "Stage 4 will answer these questions using formal experimentation "
            "and statistical inference.",
            "",
        ]
    )

    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_MD.write_text("\n".join(lines), encoding="utf-8")


def validate(
    funnel: pd.DataFrame,
    activation: pd.DataFrame,
    product_segments: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> None:
    assert funnel["count"].is_monotonic_decreasing
    assert funnel["conversion_from_previous"].between(0, 1).all()
    assert funnel["conversion_from_search"].between(0, 1).all()

    rate_cols = [
        "integration_completion_rate",
        "first_search_rate",
        "first_booking_rate",
        "activation_30d_rate",
    ]
    for col in rate_cols:
        assert activation[col].dropna().between(0, 1).all()

    for col in [
        "session_booking_conversion",
        "search_to_book_conversion",
        "no_result_rate",
        "result_selection_rate",
    ]:
        assert product_segments[col].dropna().between(0, 1).all()

    assert np.isfinite(opportunities["opportunity_score"]).all()
    assert opportunities["priority_rank"].is_unique


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    try:
        tables = load_tables(con)

        funnel = build_funnel(tables["fct_search_journey"])
        activation_segments = build_activation_segments(
            tables["stg_partners"],
            tables["mart_partner_activation"],
        )
        product_segments = build_product_segments(tables["fct_session_journey"])
        partner_growth = build_partner_growth(
            tables["mart_partner_weekly"],
            tables["mart_partner_performance"],
        )
        opportunities = build_opportunities(partner_growth)
        retention_summary = build_retention_summary(tables["mart_partner_retention"])

        validate(
            funnel,
            activation_segments,
            product_segments,
            opportunities,
        )

        outputs = {
            "mart_stage3_funnel": funnel,
            "mart_stage3_activation_segments": activation_segments,
            "mart_stage3_product_segments": product_segments,
            "mart_stage3_partner_growth": partner_growth,
            "mart_stage3_partner_opportunities": opportunities,
            "mart_stage3_retention_summary": retention_summary,
        }

        for name, frame in outputs.items():
            write_table(con, name, frame)
            frame.to_csv(OUT_DIR / f"{name}.csv", index=False)

        findings = generate_findings(
            funnel,
            activation_segments,
            product_segments,
            opportunities,
            retention_summary,
        )
        (OUT_DIR / "stage3_findings.json").write_text(
            json.dumps(findings, indent=2),
            encoding="utf-8",
        )
        write_findings_markdown(findings)

        validation_report = {
            "status": "PASS",
            "data_classification": "synthetic portfolio data",
            "tables": {name: int(len(frame)) for name, frame in outputs.items()},
            "checks": {
                "funnel_monotonic": True,
                "rates_within_bounds": True,
                "opportunity_scores_finite": True,
                "priority_rank_unique": True,
            },
        }
        (OUT_DIR / "validation_report.json").write_text(
            json.dumps(validation_report, indent=2),
            encoding="utf-8",
        )

        print()
        print("=" * 78)
        print("RAILNEXUS - STAGE 3 B2B PRODUCT GROWTH")
        print("=" * 78)
        print(f"Search funnel conversion     : {funnel.iloc[-1]['conversion_from_search']:.2%}")
        print(
            f"Largest funnel drop          : "
            f"{findings['largest_funnel_drop']['stage']} "
            f"({findings['largest_funnel_drop']['dropoff_rate']:.2%})"
        )
        print(f"Top partner opportunity      : {findings['top_partner_opportunity']['partner_id']}")
        print(
            f"Opportunity type             : "
            f"{findings['top_partner_opportunity']['opportunity_type']}"
        )
        print(f"Stage 3 exports              : {OUT_DIR.relative_to(ROOT)}")
        print(f"Executive findings           : {FINDINGS_MD.relative_to(ROOT)}")
        print("STAGE 3                     : PASS")
        print("=" * 78)

    finally:
        con.close()


if __name__ == "__main__":
    main()
