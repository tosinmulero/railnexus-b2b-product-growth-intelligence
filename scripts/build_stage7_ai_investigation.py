from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
OUT_DIR = ROOT / "reports" / "generated" / "stage7"
BRIEFS_DIR = OUT_DIR / "investigation_briefs"
DOC_PATH = ROOT / "docs" / "AI_ASSISTED_INVESTIGATION.md"
FINDINGS_MD = ROOT / "reports" / "STAGE7_INVESTIGATION_FINDINGS.md"

LOOKBACK_DAYS = 28
MIN_HISTORY_DAYS = 14
ROBUST_Z_THRESHOLD = 3.5

METRIC_RULES = {
    "session_booking_conversion": "low",
    "search_to_book_conversion": "low",
    "no_result_rate": "high",
    "avg_search_latency_ms": "high",
    "cancellation_rate": "high",
    "refund_rate": "high",
    "successful_booking_value_per_active_partner": "low",
}


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_daily_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frame = con.execute(
        """
        SELECT *
        FROM mart_daily_product_metrics
        ORDER BY metric_date
        """
    ).fetch_df()

    if frame.empty:
        raise RuntimeError("Daily product metrics are empty.")

    frame["metric_date"] = pd.to_datetime(frame["metric_date"])

    missing = [metric for metric in METRIC_RULES if metric not in frame.columns]
    if missing:
        raise RuntimeError("Required daily metrics are missing: " + ", ".join(missing))

    return frame


def robust_rolling_scores(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for metric, adverse_direction in METRIC_RULES.items():
        values = pd.to_numeric(daily[metric], errors="coerce").astype(float)

        rolling_median = (
            values.shift(1)
            .rolling(
                LOOKBACK_DAYS,
                min_periods=MIN_HISTORY_DAYS,
            )
            .median()
        )

        abs_dev = (values - rolling_median).abs()
        rolling_mad = (
            abs_dev.shift(1)
            .rolling(
                LOOKBACK_DAYS,
                min_periods=MIN_HISTORY_DAYS,
            )
            .median()
        )

        fallback_scale = float(values.std(ddof=0))
        if not np.isfinite(fallback_scale) or fallback_scale == 0:
            fallback_scale = 1e-6

        scale = (1.4826 * rolling_mad).replace(0, np.nan).fillna(fallback_scale)
        robust_z = (values - rolling_median) / scale

        for idx, date in enumerate(daily["metric_date"]):
            z = robust_z.iloc[idx]
            baseline = rolling_median.iloc[idx]
            observed = values.iloc[idx]

            if pd.isna(z) or pd.isna(baseline) or pd.isna(observed):
                continue

            adverse = (adverse_direction == "high" and z >= ROBUST_Z_THRESHOLD) or (
                adverse_direction == "low" and z <= -ROBUST_Z_THRESHOLD
            )

            if not adverse:
                continue

            delta = float(observed - baseline)
            relative_delta = (
                float(delta / baseline)
                if baseline not in (0, 0.0) and np.isfinite(baseline)
                else 0.0
            )
            abs_z = float(abs(z))

            if abs_z >= 7:
                severity = "critical"
            elif abs_z >= 5:
                severity = "high"
            else:
                severity = "medium"

            rows.append(
                {
                    "metric_date": pd.Timestamp(date),
                    "metric": metric,
                    "adverse_direction": adverse_direction,
                    "observed_value": float(observed),
                    "baseline_median": float(baseline),
                    "absolute_delta": delta,
                    "relative_delta": relative_delta,
                    "robust_z": float(z),
                    "severity": severity,
                }
            )

    anomalies = pd.DataFrame(rows)
    if anomalies.empty:
        return pd.DataFrame(
            columns=[
                "metric_date",
                "metric",
                "adverse_direction",
                "observed_value",
                "baseline_median",
                "absolute_delta",
                "relative_delta",
                "robust_z",
                "severity",
                "case_id",
            ]
        )

    severity_order = {"critical": 1, "high": 2, "medium": 3}
    anomalies["_severity_order"] = anomalies["severity"].map(severity_order)
    anomalies = anomalies.sort_values(["_severity_order", "metric_date", "metric"]).drop(
        columns="_severity_order"
    )

    anomalies["case_id"] = [
        f"RNX-{date.strftime('%Y%m%d')}-{i:03d}"
        for i, date in enumerate(anomalies["metric_date"], start=1)
    ]

    return anomalies.reset_index(drop=True)


def metric_segment_value(
    frame: pd.DataFrame,
    metric: str,
) -> float:
    if frame.empty:
        return np.nan

    if metric == "session_booking_conversion":
        return float(frame["booked_flag"].mean())

    if metric == "search_to_book_conversion":
        searches = float(frame["search_count"].sum())
        bookings = float(frame["booking_count"].sum())
        return bookings / searches if searches else np.nan

    if metric == "no_result_rate":
        searches = float(frame["search_count"].sum())
        no_result = float(frame["zero_result_searches"].sum())
        return no_result / searches if searches else np.nan

    if metric == "avg_search_latency_ms":
        searches = float(frame["search_count"].sum())
        total_latency = float(frame["total_search_latency_ms"].sum())
        return total_latency / searches if searches else np.nan

    if metric == "cancellation_rate":
        bookings = float(frame["booking_count"].sum())
        cancellations = float(frame["cancellations"].sum())
        return cancellations / bookings if bookings else np.nan

    if metric == "refund_rate":
        bookings = float(frame["booking_count"].sum())
        refunds = float(frame["refunds"].sum())
        return refunds / bookings if bookings else np.nan

    if metric == "successful_booking_value_per_active_partner":
        partners = int(frame["partner_id"].nunique())
        value = float(frame["successful_booking_value"].sum())
        return value / partners if partners else np.nan

    raise KeyError(metric)


def adverse_gap(
    current: float,
    baseline: float,
    adverse_direction: str,
) -> float:
    if not np.isfinite(current) or not np.isfinite(baseline):
        return -np.inf

    if adverse_direction == "high":
        return current - baseline
    return baseline - current


def find_primary_driver(
    con: duckdb.DuckDBPyConnection,
    anomaly: pd.Series,
) -> dict[str, object]:
    date = pd.Timestamp(anomaly["metric_date"])
    start = date - pd.Timedelta(days=LOOKBACK_DAYS)
    metric = str(anomaly["metric"])
    adverse_direction = str(anomaly["adverse_direction"])

    data = con.execute(
        """
        SELECT *
        FROM fct_session_journey
        WHERE session_date BETWEEN ? AND ?
        """,
        [start.date(), date.date()],
    ).fetch_df()

    if data.empty:
        return {
            "dimension": "none",
            "segment": "no_data",
            "current_value": None,
            "baseline_value": None,
            "adverse_gap": None,
        }

    data["session_date"] = pd.to_datetime(data["session_date"])

    dimensions = [
        "device_type",
        "traveller_type",
        "acquisition_channel",
    ]
    candidates: list[dict[str, object]] = []

    current_data = data.loc[data["session_date"] == date.normalize()]
    baseline_data = data.loc[
        (data["session_date"] >= start.normalize()) & (data["session_date"] < date.normalize())
    ]

    for dimension in dimensions:
        levels = sorted(
            set(current_data[dimension].dropna().astype(str))
            & set(baseline_data[dimension].dropna().astype(str))
        )

        for level in levels:
            current_segment = current_data.loc[current_data[dimension].astype(str) == level]
            baseline_segment = baseline_data.loc[baseline_data[dimension].astype(str) == level]

            if len(current_segment) < 5 or len(baseline_segment) < 20:
                continue

            current_value = metric_segment_value(
                current_segment,
                metric,
            )
            baseline_value = metric_segment_value(
                baseline_segment,
                metric,
            )
            gap = adverse_gap(
                current_value,
                baseline_value,
                adverse_direction,
            )

            if not np.isfinite(gap):
                continue

            candidates.append(
                {
                    "dimension": dimension,
                    "segment": level,
                    "current_value": float(current_value),
                    "baseline_value": float(baseline_value),
                    "adverse_gap": float(gap),
                    "current_sessions": int(len(current_segment)),
                    "baseline_sessions": int(len(baseline_segment)),
                }
            )

    if not candidates:
        return {
            "dimension": "none",
            "segment": "insufficient_segment_data",
            "current_value": None,
            "baseline_value": None,
            "adverse_gap": None,
        }

    return max(candidates, key=lambda item: item["adverse_gap"])


def top_partner_context(
    con: duckdb.DuckDBPyConnection,
    anomaly: pd.Series,
) -> list[dict[str, object]]:
    date = pd.Timestamp(anomaly["metric_date"]).date()

    frame = con.execute(
        """
        SELECT
            partner_id,
            COUNT(*) AS sessions,
            SUM(search_count) AS searches,
            SUM(booking_count) AS bookings,
            AVG(booked_flag) AS session_booking_conversion,
            SUM(zero_result_searches)::DOUBLE
                / NULLIF(SUM(search_count), 0) AS no_result_rate,
            SUM(successful_booking_value) AS successful_booking_value
        FROM fct_session_journey
        WHERE session_date = ?
        GROUP BY partner_id
        ORDER BY sessions DESC, partner_id
        LIMIT 5
        """,
        [date],
    ).fetch_df()

    return [
        {
            "partner_id": str(row.partner_id),
            "sessions": int(row.sessions),
            "searches": int(row.searches),
            "bookings": int(row.bookings),
            "session_booking_conversion": float(row.session_booking_conversion),
            "no_result_rate": (float(row.no_result_rate) if pd.notna(row.no_result_rate) else None),
            "successful_booking_value": float(row.successful_booking_value),
        }
        for row in frame.itertuples(index=False)
    ]


def top_route_context(
    con: duckdb.DuckDBPyConnection,
    anomaly: pd.Series,
) -> list[dict[str, object]]:
    date = pd.Timestamp(anomaly["metric_date"]).date()

    frame = con.execute(
        """
        SELECT
            origin_station_id,
            destination_station_id,
            COUNT(*) AS searches,
            SUM(booked_flag) AS booked_searches,
            AVG(no_result_flag) AS no_result_rate,
            AVG(search_latency_ms) AS avg_search_latency_ms
        FROM fct_search_journey
        WHERE search_date = ?
        GROUP BY origin_station_id, destination_station_id
        ORDER BY searches DESC, origin_station_id, destination_station_id
        LIMIT 5
        """,
        [date],
    ).fetch_df()

    return [
        {
            "origin_station_id": str(row.origin_station_id),
            "destination_station_id": str(row.destination_station_id),
            "searches": int(row.searches),
            "booked_searches": int(row.booked_searches),
            "no_result_rate": float(row.no_result_rate),
            "avg_search_latency_ms": float(row.avg_search_latency_ms),
        }
        for row in frame.itertuples(index=False)
    ]


def hypotheses_for_metric(metric: str) -> list[str]:
    mapping = {
        "session_booking_conversion": [
            (
                "Check whether the drop is concentrated by device, "
                "acquisition channel or traveller type."
            ),
            "Compare no-result rate and search latency on the same date.",
            "Inspect high-volume partners for simultaneous conversion deterioration.",
        ],
        "search_to_book_conversion": [
            "Check route-level booking conversion and no-result exposure.",
            "Inspect whether journey-result selection weakened before booking.",
            "Compare high-volume partner performance with the prior 28-day baseline.",
        ],
        "no_result_rate": [
            "Inspect origin-destination routes contributing the most zero-result searches.",
            "Check whether the issue is concentrated by country, partner or device.",
            "Compare latency and booking conversion for the same affected segments.",
        ],
        "avg_search_latency_ms": [
            "Check whether latency is concentrated by device or acquisition channel.",
            "Compare latency with conversion and no-result changes.",
            "Review high-volume routes and partners for shared degradation.",
        ],
        "cancellation_rate": [
            "Check whether the increase is concentrated in particular partners or journey types.",
            "Compare booking-value mix and route mix with recent history.",
            "Treat the signal cautiously on low-booking-volume days.",
        ],
        "refund_rate": [
            "Check whether refunds follow a cancellation spike or specific partner/route mix.",
            "Compare booking volume to ensure the rate is not driven by a small denominator.",
            "Inspect journey type and commercial-value mix.",
        ],
        "successful_booking_value_per_active_partner": [
            "Separate traffic loss from conversion loss and lower booking value.",
            "Inspect whether high-value partners contributed less than normal.",
            "Compare partner and route mix with the prior 28-day baseline.",
        ],
    }
    return mapping.get(metric, ["Review the supporting product metrics and segment mix."])


def owner_for_metric(metric: str) -> str:
    if metric in {"no_result_rate", "avg_search_latency_ms"}:
        return "Search / Supply Product"
    if metric in {"cancellation_rate", "refund_rate"}:
        return "Booking / Post-Booking Product"
    if metric == "successful_booking_value_per_active_partner":
        return "B2B Growth / Commercial"
    return "Product Analytics / Growth"


def build_cases(
    con: duckdb.DuckDBPyConnection,
    anomalies: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    packets: list[dict[str, object]] = []

    for _, anomaly in anomalies.iterrows():
        driver = find_primary_driver(con, anomaly)
        partners = top_partner_context(con, anomaly)
        routes = top_route_context(con, anomaly)
        metric = str(anomaly["metric"])

        packet = {
            "case_id": str(anomaly["case_id"]),
            "data_classification": "synthetic portfolio data",
            "metric_date": pd.Timestamp(anomaly["metric_date"]).date().isoformat(),
            "metric": metric,
            "severity": str(anomaly["severity"]),
            "observation": {
                "observed_value": float(anomaly["observed_value"]),
                "baseline_median": float(anomaly["baseline_median"]),
                "absolute_delta": float(anomaly["absolute_delta"]),
                "relative_delta": float(anomaly["relative_delta"]),
                "robust_z": float(anomaly["robust_z"]),
            },
            "primary_segment_driver": driver,
            "top_partner_context": partners,
            "top_route_context": routes,
            "hypotheses_to_test": hypotheses_for_metric(metric),
            "recommended_owner": owner_for_metric(metric),
            "human_review_required": True,
            "reasoning_policy": {
                "observations_are_facts": True,
                "hypotheses_are_not_causal_claims": True,
                "do_not_invent_missing_evidence": True,
                "do_not_present_synthetic_values_as_trainline_results": True,
            },
        }
        packets.append(packet)

        rows.append(
            {
                "case_id": packet["case_id"],
                "metric_date": pd.Timestamp(anomaly["metric_date"]),
                "metric": metric,
                "severity": str(anomaly["severity"]),
                "robust_z": float(anomaly["robust_z"]),
                "observed_value": float(anomaly["observed_value"]),
                "baseline_median": float(anomaly["baseline_median"]),
                "primary_driver_dimension": str(driver["dimension"]),
                "primary_driver_segment": str(driver["segment"]),
                "recommended_owner": packet["recommended_owner"],
                "human_review_required": True,
            }
        )

    return pd.DataFrame(rows), packets


def write_case_briefs(packets: list[dict[str, object]]) -> None:
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)

    for packet in packets:
        obs = packet["observation"]
        driver = packet["primary_segment_driver"]

        lines = [
            f"# Investigation Brief â€” {packet['case_id']}",
            "",
            "> Synthetic portfolio data. Human review required.",
            "",
            "## Observation",
            "",
            f"- Date: **{packet['metric_date']}**",
            f"- Metric: **{packet['metric']}**",
            f"- Severity: **{packet['severity']}**",
            f"- Observed value: **{obs['observed_value']:.6f}**",
            f"- Prior baseline median: **{obs['baseline_median']:.6f}**",
            f"- Relative delta: **{obs['relative_delta']:.2%}**",
            f"- Robust z-score: **{obs['robust_z']:.2f}**",
            "",
            "## Strongest Observed Segment Driver",
            "",
            f"- Dimension: **{driver['dimension']}**",
            f"- Segment: **{driver['segment']}**",
            "",
            "## Hypotheses to Test",
            "",
        ]

        for hypothesis in packet["hypotheses_to_test"]:
            lines.append(f"- {hypothesis}")

        lines.extend(
            [
                "",
                "## Recommended Owner",
                "",
                f"**{packet['recommended_owner']}**",
                "",
                "## Responsible-AI Guardrails",
                "",
                "- Keep observations separate from hypotheses.",
                "- Do not infer causality from correlation or anomaly detection.",
                "- Do not invent evidence that is absent from the packet.",
                "- Require human review before escalation or product action.",
                "- Never present these synthetic values as actual Trainline results.",
                "",
            ]
        )

        path = BRIEFS_DIR / f"{packet['case_id']}.md"
        path.write_text("\n".join(lines), encoding="utf-8")


def write_prompt_pack(packets: list[dict[str, object]]) -> None:
    prompt = """# RailNexus AI Investigation Prompt

You are assisting a product data scientist with anomaly investigation.

Use only the supplied evidence packet.

Rules:
1. Separate OBSERVATIONS from HYPOTHESES.
2. Never claim causality unless the evidence directly supports it.
3. Never invent metrics, dimensions, partners, routes or business impact.
4. Rank plausible next checks by expected diagnostic value.
5. State when evidence is insufficient.
6. Treat all RailNexus values as synthetic portfolio data.
7. Require human review before any product or commercial escalation.

Return:
- concise incident summary;
- strongest evidence;
- plausible hypotheses;
- recommended diagnostic checks;
- stakeholder owner;
- uncertainty / limitations.
"""
    (OUT_DIR / "ai_investigation_prompt.md").write_text(
        prompt,
        encoding="utf-8",
    )

    packet_path = OUT_DIR / "ai_context_packets.json"
    packet_path.write_text(
        json.dumps(
            packets,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )


def write_table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    frame: pd.DataFrame,
) -> None:
    temp = f"_stage7_{name}"
    con.register(temp, frame)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "{temp}"')
    finally:
        con.unregister(temp)


def write_documentation(
    anomalies: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    method = f"""# RailNexus â€” AI-Assisted Investigation & Anomaly Triage

## Purpose

Stage 7 automates recurring product investigation preparation.

It does **not** allow an AI system to make autonomous product or commercial
decisions.

## Detection method

Daily product metrics are compared with a trailing **{LOOKBACK_DAYS}-day**
baseline using only prior observations.

A minimum of **{MIN_HISTORY_DAYS} prior days** is required.

Robust z-scores use a rolling median and median absolute deviation.

Adverse anomalies are surfaced at an absolute robust z-score of at least
**{ROBUST_Z_THRESHOLD}**.

## Metrics monitored

- session booking conversion;
- search-to-book conversion;
- no-result rate;
- search latency;
- cancellation rate;
- refund rate;
- successful booking value per active partner.

## Investigation enrichment

Each case receives:

- strongest observed segment driver;
- high-volume partner context;
- high-volume route context;
- metric-specific hypotheses to test;
- recommended functional owner;
- an evidence packet suitable for constrained LLM assistance.

## Responsible AI controls

- observations and hypotheses are separated;
- no causal claims are generated by the anomaly engine;
- no missing evidence may be invented;
- synthetic values must not be represented as Trainline results;
- every case explicitly requires human review;
- the core pipeline works without sending data to an external AI provider.

## Current run

- anomalies detected: **{len(anomalies)}**
- investigation cases created: **{len(cases)}**

The prompt pack demonstrates how generative AI can accelerate investigation
summarisation while statistical validation and decision ownership remain
human-controlled.
"""
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(method, encoding="utf-8")

    if cases.empty:
        findings = """# RailNexus â€” Stage 7 Investigation Findings

> Synthetic portfolio data.

No daily metric crossed the configured adverse anomaly threshold in this
validation run. The triage framework, evidence packet generation and
responsible-AI controls were still successfully validated.
"""
    else:
        severity_counts = cases["severity"].value_counts().to_dict()
        top = cases.iloc[0]
        findings = f"""# RailNexus â€” Stage 7 Investigation Findings

> All values are synthetic portfolio data.

## Detection Summary

- Investigation cases: **{len(cases)}**
- Critical: **{int(severity_counts.get("critical", 0))}**
- High: **{int(severity_counts.get("high", 0))}**
- Medium: **{int(severity_counts.get("medium", 0))}**

## Highest-Priority Case

- Case: **{top["case_id"]}**
- Date: **{pd.Timestamp(top["metric_date"]).date()}**
- Metric: **{top["metric"]}**
- Severity: **{top["severity"]}**
- Robust z-score: **{float(top["robust_z"]):.2f}**
- Primary observed driver: **{top["primary_driver_dimension"]} = {top["primary_driver_segment"]}**
- Recommended owner: **{top["recommended_owner"]}**

## AI-Assisted Workflow

The project creates structured evidence packets and a constrained investigation
prompt so an AI assistant can accelerate summarisation and next-step planning
without inventing evidence or making autonomous decisions.

Every escalation remains subject to human review.
"""

    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_MD.write_text(findings, encoding="utf-8")


def validate_outputs(
    anomalies: pd.DataFrame,
    cases: pd.DataFrame,
    packets: list[dict[str, object]],
) -> dict[str, bool]:
    checks = {
        "case_packet_count_matches": len(cases) == len(packets),
        "case_ids_unique": (True if cases.empty else cases["case_id"].is_unique),
        "human_review_required": (
            True if cases.empty else bool(cases["human_review_required"].all())
        ),
        "anomaly_scores_finite": (
            True if anomalies.empty else bool(np.isfinite(anomalies["robust_z"]).all())
        ),
        "prompt_pack_exists": (OUT_DIR / "ai_investigation_prompt.md").exists(),
        "context_packets_exist": (OUT_DIR / "ai_context_packets.json").exists(),
    }

    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise RuntimeError("Stage 7 validation failed: " + ", ".join(failed))

    return {name: bool(value) for name, value in checks.items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    try:
        daily = load_daily_metrics(con)
        anomalies = robust_rolling_scores(daily)
        cases, packets = build_cases(con, anomalies)

        anomalies.to_csv(
            OUT_DIR / "daily_metric_anomalies.csv",
            index=False,
        )
        cases.to_csv(
            OUT_DIR / "investigation_case_register.csv",
            index=False,
        )

        write_case_briefs(packets)
        write_prompt_pack(packets)
        write_documentation(anomalies, cases)

        write_table(
            con,
            "mart_stage7_daily_anomalies",
            anomalies,
        )
        write_table(
            con,
            "mart_stage7_investigation_cases",
            cases,
        )

        checks = validate_outputs(
            anomalies,
            cases,
            packets,
        )

        validation = {
            "status": "PASS",
            "data_classification": "synthetic portfolio data",
            "lookback_days": LOOKBACK_DAYS,
            "minimum_history_days": MIN_HISTORY_DAYS,
            "robust_z_threshold": ROBUST_Z_THRESHOLD,
            "anomalies_detected": int(len(anomalies)),
            "cases_created": int(len(cases)),
            "checks": checks,
        }
        (OUT_DIR / "validation_report.json").write_text(
            json.dumps(
                validation,
                indent=2,
                default=json_default,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 78)
        print("RAILNEXUS - STAGE 7 AI-ASSISTED INVESTIGATION")
        print("=" * 78)
        print(f"Daily metrics monitored       : {len(METRIC_RULES)}")
        print(f"Anomalies detected            : {len(anomalies):,}")
        print(f"Investigation cases created   : {len(cases):,}")
        print(
            f"AI context packets            : "
            f"{(OUT_DIR / 'ai_context_packets.json').relative_to(ROOT)}"
        )
        print(
            f"Investigation prompt          : "
            f"{(OUT_DIR / 'ai_investigation_prompt.md').relative_to(ROOT)}"
        )
        print(f"Responsible-AI method         : {DOC_PATH.relative_to(ROOT)}")
        print(f"Executive findings            : {FINDINGS_MD.relative_to(ROOT)}")
        print("STAGE 7                       : PASS")
        print("=" * 78)

    finally:
        con.close()


if __name__ == "__main__":
    main()
