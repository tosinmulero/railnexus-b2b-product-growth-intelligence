from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_effectsize,
    proportions_ztest,
)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
OUT_DIR = ROOT / "reports" / "generated" / "stage4"
FINDINGS_MD = ROOT / "reports" / "STAGE4_EXPERIMENT_FINDINGS.md"
EXPERIMENT_CARD = ROOT / "docs" / "EXPERIMENT_CARD.md"

ALPHA = 0.05
TARGET_POWER = 0.80
EXPERIMENT_NAME = "smart_alternatives_v1"


def scalar_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def normal_ci(diff: float, se: float, alpha: float = ALPHA) -> tuple[float, float]:
    z = stats.norm.ppf(1 - alpha / 2)
    return diff - z * se, diff + z * se


def diff_in_proportions(
    treatment_success: int,
    treatment_n: int,
    control_success: int,
    control_n: int,
) -> dict[str, float]:
    p_t = scalar_div(treatment_success, treatment_n)
    p_c = scalar_div(control_success, control_n)
    diff = p_t - p_c

    se_unpooled = np.sqrt(
        scalar_div(p_t * (1 - p_t), treatment_n) + scalar_div(p_c * (1 - p_c), control_n)
    )
    ci_low, ci_high = normal_ci(diff, se_unpooled)

    z_stat, p_value = proportions_ztest(
        count=np.array([treatment_success, control_success]),
        nobs=np.array([treatment_n, control_n]),
        alternative="two-sided",
    )

    return {
        "control_rate": p_c,
        "variant_rate": p_t,
        "absolute_effect": diff,
        "relative_effect": scalar_div(diff, p_c),
        "standard_error": float(se_unpooled),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
    }


def load_experiment_tables(
    con: duckdb.DuckDBPyConnection,
) -> dict[str, pd.DataFrame]:
    session = con.execute(
        """
        SELECT *
        FROM fct_session_journey
        WHERE experiment_name = ?
          AND experiment_variant IN ('control', 'variant')
        """,
        [EXPERIMENT_NAME],
    ).fetch_df()

    searches = con.execute(
        """
        SELECT *
        FROM fct_search_journey
        WHERE experiment_variant IN ('control', 'variant')
          AND search_ts >= TIMESTAMP '2026-04-01'
          AND search_ts < TIMESTAMP '2026-05-16'
        """
    ).fetch_df()

    bookings = con.execute(
        """
        SELECT *
        FROM stg_bookings
        WHERE experiment_variant IN ('control', 'variant')
          AND booking_ts >= TIMESTAMP '2026-04-01'
          AND booking_ts < TIMESTAMP '2026-05-16'
        """
    ).fetch_df()

    preperiod = con.execute(
        """
        SELECT *
        FROM fct_session_journey
        WHERE session_ts < TIMESTAMP '2026-04-01'
        """
    ).fetch_df()

    partners = con.execute(
        """
        SELECT
            partner_id,
            partner_type,
            country,
            integration_type,
            contract_tier
        FROM stg_partners
        """
    ).fetch_df()

    if session.empty:
        raise RuntimeError("No experiment sessions found.")

    return {
        "session": session,
        "searches": searches,
        "bookings": bookings,
        "preperiod": preperiod,
        "partners": partners,
    }


def sample_ratio_mismatch(exp: pd.DataFrame) -> pd.DataFrame:
    counts = exp["experiment_variant"].value_counts().reindex(["control", "variant"], fill_value=0)

    observed = counts.to_numpy(dtype=float)
    expected = np.repeat(observed.sum() / 2, 2)
    chi2, p_value = stats.chisquare(observed, f_exp=expected)

    return pd.DataFrame(
        [
            {
                "control_sessions": int(counts["control"]),
                "variant_sessions": int(counts["variant"]),
                "variant_share": scalar_div(
                    float(counts["variant"]),
                    float(counts.sum()),
                ),
                "chi_square": float(chi2),
                "p_value": float(p_value),
                "srm_flag": bool(p_value < 0.001),
            }
        ]
    )


def build_balance_table(exp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    categorical = [
        "device_type",
        "traveller_type",
        "acquisition_channel",
        "is_logged_in",
    ]

    for col in categorical:
        ctab = pd.crosstab(
            exp[col],
            exp["experiment_variant"],
        ).reindex(columns=["control", "variant"], fill_value=0)

        for level, row in ctab.iterrows():
            control_n = int(row["control"])
            variant_n = int(row["variant"])
            total_control = int((exp["experiment_variant"] == "control").sum())
            total_variant = int((exp["experiment_variant"] == "variant").sum())
            control_share = scalar_div(control_n, total_control)
            variant_share = scalar_div(variant_n, total_variant)

            pooled = scalar_div(
                control_n + variant_n,
                total_control + total_variant,
            )
            denom = np.sqrt(max(pooled * (1 - pooled), 1e-12))
            smd = (variant_share - control_share) / denom

            rows.append(
                {
                    "covariate": col,
                    "level": str(level),
                    "control_share": control_share,
                    "variant_share": variant_share,
                    "standardized_difference": float(smd),
                    "abs_standardized_difference": float(abs(smd)),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["abs_standardized_difference", "covariate"],
        ascending=[False, True],
    )


def build_primary_effect(exp: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        exp.groupby("experiment_variant", as_index=False)
        .agg(
            sessions=("session_id", "count"),
            booking_sessions=("booked_flag", "sum"),
            successful_booking_value=("successful_booking_value", "sum"),
        )
        .set_index("experiment_variant")
    )

    control = grouped.loc["control"]
    variant = grouped.loc["variant"]

    effect = diff_in_proportions(
        treatment_success=int(variant["booking_sessions"]),
        treatment_n=int(variant["sessions"]),
        control_success=int(control["booking_sessions"]),
        control_n=int(control["sessions"]),
    )

    effect.update(
        {
            "metric": "session_booking_conversion",
            "control_n": int(control["sessions"]),
            "variant_n": int(variant["sessions"]),
            "control_successes": int(control["booking_sessions"]),
            "variant_successes": int(variant["booking_sessions"]),
        }
    )

    return pd.DataFrame([effect])


def build_preperiod_partner_baseline(
    preperiod: pd.DataFrame,
) -> pd.DataFrame:
    baseline = preperiod.groupby("partner_id", as_index=False).agg(
        pre_sessions=("session_id", "count"),
        pre_booking_sessions=("booked_flag", "sum"),
    )
    baseline["pre_partner_conversion"] = np.where(
        baseline["pre_sessions"] > 0,
        baseline["pre_booking_sessions"] / baseline["pre_sessions"],
        0.0,
    )
    return baseline[["partner_id", "pre_sessions", "pre_partner_conversion"]]


def build_cuped_effect(
    exp: pd.DataFrame,
    preperiod: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = build_preperiod_partner_baseline(preperiod)
    data = exp.merge(
        baseline,
        on="partner_id",
        how="left",
        validate="many_to_one",
    )

    global_pre = scalar_div(
        float(preperiod["booked_flag"].sum()),
        float(len(preperiod)),
    )
    data["pre_partner_conversion"] = data["pre_partner_conversion"].fillna(global_pre)
    data["pre_sessions"] = data["pre_sessions"].fillna(0)

    x = data["pre_partner_conversion"].astype(float)
    y = data["booked_flag"].astype(float)

    var_x = float(np.var(x, ddof=1))
    theta = float(np.cov(y, x, ddof=1)[0, 1] / var_x) if var_x > 0 else 0.0

    centered_x = x - float(x.mean())
    data["cuped_outcome"] = y - theta * centered_x

    control = data.loc[
        data["experiment_variant"] == "control",
        "cuped_outcome",
    ]
    variant = data.loc[
        data["experiment_variant"] == "variant",
        "cuped_outcome",
    ]

    diff = float(variant.mean() - control.mean())
    se = float(np.sqrt(variant.var(ddof=1) / len(variant) + control.var(ddof=1) / len(control)))
    ci_low, ci_high = normal_ci(diff, se)
    t_stat, p_value = stats.ttest_ind(
        variant,
        control,
        equal_var=False,
    )

    raw_variance = float(y.var(ddof=1))
    cuped_variance = float(data["cuped_outcome"].var(ddof=1))
    variance_reduction = 1 - scalar_div(cuped_variance, raw_variance) if raw_variance > 0 else 0.0

    result = pd.DataFrame(
        [
            {
                "metric": "session_booking_conversion",
                "method": "CUPED-style partner pre-period adjustment",
                "theta": theta,
                "control_adjusted_mean": float(control.mean()),
                "variant_adjusted_mean": float(variant.mean()),
                "adjusted_effect": diff,
                "standard_error": se,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "variance_reduction": variance_reduction,
            }
        ]
    )

    return result, data


def build_cluster_adjusted_effect(
    cuped_data: pd.DataFrame,
) -> pd.DataFrame:
    data = cuped_data.copy()
    data["treatment"] = (data["experiment_variant"] == "variant").astype(int)

    formula = (
        "booked_flag ~ treatment + pre_partner_conversion "
        "+ C(device_type) + C(traveller_type) "
        "+ C(acquisition_channel) + is_logged_in"
    )

    model = smf.ols(
        formula=formula,
        data=data,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["partner_id"]},
    )

    effect = float(model.params["treatment"])
    se = float(model.bse["treatment"])
    p_value = float(model.pvalues["treatment"])
    ci_low, ci_high = model.conf_int().loc["treatment"].astype(float)

    return pd.DataFrame(
        [
            {
                "method": "cluster-robust covariate-adjusted LPM",
                "adjusted_effect": effect,
                "standard_error": se,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": p_value,
                "partner_clusters": int(data["partner_id"].nunique()),
                "observations": int(len(data)),
                "r_squared": float(model.rsquared),
            }
        ]
    )


def build_guardrails(
    searches: pd.DataFrame,
    bookings: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    search_group = searches.groupby("experiment_variant")

    control_search = search_group.get_group("control")
    variant_search = search_group.get_group("variant")

    no_result = diff_in_proportions(
        treatment_success=int(variant_search["no_result_flag"].sum()),
        treatment_n=int(len(variant_search)),
        control_success=int(control_search["no_result_flag"].sum()),
        control_n=int(len(control_search)),
    )
    rows.append(
        {
            "metric": "no_result_rate",
            "metric_type": "binary",
            **no_result,
        }
    )

    t_stat, latency_p = stats.ttest_ind(
        variant_search["search_latency_ms"],
        control_search["search_latency_ms"],
        equal_var=False,
    )
    latency_diff = float(
        variant_search["search_latency_ms"].mean() - control_search["search_latency_ms"].mean()
    )
    latency_se = float(
        np.sqrt(
            variant_search["search_latency_ms"].var(ddof=1) / len(variant_search)
            + control_search["search_latency_ms"].var(ddof=1) / len(control_search)
        )
    )
    latency_low, latency_high = normal_ci(
        latency_diff,
        latency_se,
    )
    rows.append(
        {
            "metric": "search_latency_ms",
            "metric_type": "continuous",
            "control_rate": float(control_search["search_latency_ms"].mean()),
            "variant_rate": float(variant_search["search_latency_ms"].mean()),
            "absolute_effect": latency_diff,
            "relative_effect": scalar_div(
                latency_diff,
                float(control_search["search_latency_ms"].mean()),
            ),
            "standard_error": latency_se,
            "ci_low": float(latency_low),
            "ci_high": float(latency_high),
            "z_stat": float(t_stat),
            "p_value": float(latency_p),
        }
    )

    if not bookings.empty:
        booking_group = bookings.groupby("experiment_variant")

        if {"control", "variant"}.issubset(set(bookings["experiment_variant"])):
            control_booking = booking_group.get_group("control")
            variant_booking = booking_group.get_group("variant")

            for metric, flag in [
                ("cancellation_rate", "cancelled_flag"),
                ("refund_rate", "refund_flag"),
            ]:
                effect = diff_in_proportions(
                    treatment_success=int(variant_booking[flag].sum()),
                    treatment_n=int(len(variant_booking)),
                    control_success=int(control_booking[flag].sum()),
                    control_n=int(len(control_booking)),
                )
                rows.append(
                    {
                        "metric": metric,
                        "metric_type": "binary",
                        **effect,
                    }
                )

    guardrails = pd.DataFrame(rows)
    guardrails["significant_at_0_05"] = guardrails["p_value"] < ALPHA
    return guardrails


def build_segment_effects(
    exp: pd.DataFrame,
    partners: pd.DataFrame,
) -> pd.DataFrame:
    data = exp.merge(
        partners,
        on="partner_id",
        how="left",
        validate="many_to_one",
    )

    dimensions = [
        "device_type",
        "traveller_type",
        "partner_type",
        "integration_type",
        "contract_tier",
    ]

    rows: list[dict[str, object]] = []

    for dimension in dimensions:
        for level, subset in data.groupby(dimension):
            counts = subset.groupby("experiment_variant").agg(
                n=("session_id", "count"),
                successes=("booked_flag", "sum"),
            )

            if not {"control", "variant"}.issubset(counts.index):
                continue

            if counts.loc["control", "n"] < 50:
                continue
            if counts.loc["variant", "n"] < 50:
                continue

            effect = diff_in_proportions(
                treatment_success=int(counts.loc["variant", "successes"]),
                treatment_n=int(counts.loc["variant", "n"]),
                control_success=int(counts.loc["control", "successes"]),
                control_n=int(counts.loc["control", "n"]),
            )

            rows.append(
                {
                    "dimension": dimension,
                    "segment": str(level),
                    "control_n": int(counts.loc["control", "n"]),
                    "variant_n": int(counts.loc["variant", "n"]),
                    **effect,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    reject, adjusted_p, _, _ = multipletests(
        out["p_value"].to_numpy(),
        alpha=ALPHA,
        method="fdr_bh",
    )
    out["p_value_fdr_bh"] = adjusted_p
    out["significant_after_fdr"] = reject

    return out.sort_values(["p_value_fdr_bh", "dimension", "segment"]).reset_index(drop=True)


def build_power_mde(primary: pd.DataFrame) -> dict[str, float]:
    row = primary.iloc[0]
    p_control = float(row["control_rate"])
    n_control = int(row["control_n"])
    n_variant = int(row["variant_n"])
    ratio = scalar_div(n_variant, n_control)

    power_model = NormalIndPower()
    observed_effect_size = abs(
        proportion_effectsize(
            float(row["variant_rate"]),
            p_control,
        )
    )

    observed_power = float(
        power_model.power(
            effect_size=observed_effect_size,
            nobs1=n_control,
            alpha=ALPHA,
            ratio=ratio,
            alternative="two-sided",
        )
    )

    min_h = float(
        power_model.solve_power(
            effect_size=None,
            nobs1=n_control,
            alpha=ALPHA,
            power=TARGET_POWER,
            ratio=ratio,
            alternative="two-sided",
        )
    )

    baseline_angle = np.arcsin(np.sqrt(p_control))
    p2 = np.sin(baseline_angle + min_h / 2) ** 2
    mde_absolute = float(abs(p2 - p_control))

    return {
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "control_n": n_control,
        "variant_n": n_variant,
        "allocation_ratio_variant_to_control": ratio,
        "control_baseline_rate": p_control,
        "observed_absolute_effect": float(row["absolute_effect"]),
        "observed_standardized_effect_size": observed_effect_size,
        "observed_approximate_power": observed_power,
        "minimum_detectable_absolute_effect": mde_absolute,
    }


def build_business_impact(
    primary: pd.DataFrame,
    exp: pd.DataFrame,
) -> pd.DataFrame:
    row = primary.iloc[0]
    variant_n = int(row["variant_n"])
    effect = float(row["absolute_effect"])

    control_booked = exp.loc[(exp["experiment_variant"] == "control") & (exp["booked_flag"] == 1)]

    average_successful_value = (
        float(control_booked["successful_booking_value"].mean())
        if not control_booked.empty
        else 0.0
    )

    incremental_booking_sessions = effect * variant_n
    estimated_incremental_successful_value = incremental_booking_sessions * average_successful_value

    return pd.DataFrame(
        [
            {
                "variant_sessions": variant_n,
                "raw_absolute_conversion_effect": effect,
                "estimated_incremental_booking_sessions": (incremental_booking_sessions),
                "control_avg_successful_value_per_booked_session": (average_successful_value),
                "estimated_incremental_successful_booking_value": (
                    estimated_incremental_successful_value
                ),
                "interpretation": (
                    "Synthetic scenario estimate only; not actual Trainline financial impact."
                ),
            }
        ]
    )


def write_table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    frame: pd.DataFrame,
) -> None:
    temp = f"_stage4_{name}"
    con.register(temp, frame)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "{temp}"')
    finally:
        con.unregister(temp)


def write_experiment_card(
    srm: pd.DataFrame,
    primary: pd.DataFrame,
    power: dict[str, float],
) -> None:
    srm_row = srm.iloc[0]
    primary_row = primary.iloc[0]

    text = f"""# RailNexus Experiment Card

## Experiment

**Smart Journey Alternatives v1**

## Data classification

All experiment, partner, traveller and commercial values are synthetic
portfolio data. No proprietary Trainline data is used.

## Experimental unit

Session-level randomisation.

A session is independently assigned to control or variant during the
experiment window.

## Analysis population

Intent-to-treat sessions assigned to either control or variant between
1 April 2026 and 15 May 2026.

## Hypothesis

Smart Journey Alternatives reduces journey-choice friction and increases
session booking conversion without materially worsening customer-experience
guardrails.

## Primary metric

**Session booking conversion**

Numerator: experiment sessions with at least one booking.

Denominator: all eligible experiment sessions.

## Guardrails

- no-result rate;
- search latency;
- cancellation rate;
- refund rate.

## Randomisation health

- Control sessions: {int(srm_row["control_sessions"]):,}
- Variant sessions: {int(srm_row["variant_sessions"]):,}
- Variant share: {float(srm_row["variant_share"]):.2%}
- SRM p-value: {float(srm_row["p_value"]):.6f}
- SRM flag: {bool(srm_row["srm_flag"])}

## Primary effect

- Control conversion: {float(primary_row["control_rate"]):.2%}
- Variant conversion: {float(primary_row["variant_rate"]):.2%}
- Absolute effect: {float(primary_row["absolute_effect"]):.2%}
- 95% CI: [{float(primary_row["ci_low"]):.2%}, {float(primary_row["ci_high"]):.2%}]
- p-value: {float(primary_row["p_value"]):.6f}

## Power

- Target power: {power["target_power"]:.0%}
- Approximate observed power: {power["observed_approximate_power"]:.2%}
- Approximate MDE: {power["minimum_detectable_absolute_effect"]:.2%}

## Covariate adjustment

The analysis uses a pre-experiment partner booking-conversion covariate for
CUPED-style variance reduction and a cluster-robust linear probability model
with partner-level clustered standard errors.

Only pre-treatment partner history is used for this adjustment.

## Heterogeneous treatment effects

Pre-specified descriptive segments include device type, traveller type,
partner type, integration type and contract tier.

Benjamini-Hochberg false-discovery-rate correction is applied across segment
tests. Segment results are exploratory and must not override the primary
experiment decision.

## Decision rule

A positive product decision should require:

1. no sample-ratio-mismatch concern;
2. a primary effect whose uncertainty is compatible with practical value;
3. no unacceptable deterioration in guardrails;
4. sufficient power for the effect size of interest;
5. no reliance on post-treatment covariates.

The project does not claim real Trainline impact.
"""
    EXPERIMENT_CARD.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_CARD.write_text(text, encoding="utf-8")


def write_findings(
    primary: pd.DataFrame,
    cuped: pd.DataFrame,
    adjusted: pd.DataFrame,
    srm: pd.DataFrame,
    guardrails: pd.DataFrame,
    power: dict[str, float],
    segments: pd.DataFrame,
    impact: pd.DataFrame,
) -> dict[str, object]:
    p = primary.iloc[0]
    c = cuped.iloc[0]
    a = adjusted.iloc[0]
    s = srm.iloc[0]
    i = impact.iloc[0]

    adverse_guardrails = guardrails.loc[
        (guardrails["significant_at_0_05"])
        & (guardrails["absolute_effect"] > 0)
        & guardrails["metric"].isin(
            [
                "no_result_rate",
                "search_latency_ms",
                "cancellation_rate",
                "refund_rate",
            ]
        )
    ]

    significant_segments = (
        segments.loc[segments["significant_after_fdr"]] if not segments.empty else segments
    )

    findings = {
        "data_classification": "synthetic portfolio data",
        "sample_ratio_mismatch": {
            "p_value": float(s["p_value"]),
            "flag": bool(s["srm_flag"]),
        },
        "primary_effect": {
            "control_rate": float(p["control_rate"]),
            "variant_rate": float(p["variant_rate"]),
            "absolute_effect": float(p["absolute_effect"]),
            "relative_effect": float(p["relative_effect"]),
            "ci_low": float(p["ci_low"]),
            "ci_high": float(p["ci_high"]),
            "p_value": float(p["p_value"]),
        },
        "cuped_effect": {
            "adjusted_effect": float(c["adjusted_effect"]),
            "ci_low": float(c["ci_low"]),
            "ci_high": float(c["ci_high"]),
            "p_value": float(c["p_value"]),
            "variance_reduction": float(c["variance_reduction"]),
        },
        "cluster_adjusted_effect": {
            "adjusted_effect": float(a["adjusted_effect"]),
            "ci_low": float(a["ci_low"]),
            "ci_high": float(a["ci_high"]),
            "p_value": float(a["p_value"]),
        },
        "power": power,
        "guardrails": {
            "adverse_significant_count": int(len(adverse_guardrails)),
        },
        "heterogeneity": {
            "segments_tested": int(len(segments)),
            "segments_significant_after_fdr": int(len(significant_segments)),
        },
        "synthetic_business_impact": {
            "estimated_incremental_booking_sessions": float(
                i["estimated_incremental_booking_sessions"]
            ),
            "estimated_incremental_successful_booking_value": float(
                i["estimated_incremental_successful_booking_value"]
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stage4_findings.json").write_text(
        json.dumps(findings, indent=2),
        encoding="utf-8",
    )

    if bool(s["srm_flag"]):
        decision = "Do not interpret treatment effects until the sample-ratio mismatch is resolved."
    elif float(p["p_value"]) < ALPHA and len(adverse_guardrails) == 0:
        decision = (
            "The synthetic experiment supports a positive product signal "
            "on the primary metric with no statistically detected adverse "
            "guardrail movement under the Stage 4 rules."
        )
    elif float(p["p_value"]) < ALPHA:
        decision = (
            "The primary metric shows a statistically detectable signal, "
            "but at least one guardrail requires product review before a "
            "launch decision."
        )
    else:
        decision = (
            "The synthetic experiment does not provide sufficient evidence "
            "for a positive launch decision at the 5% significance level."
        )

    lines = [
        "# RailNexus — Stage 4 Experiment Findings",
        "",
        "> All values are synthetic portfolio results. They are not actual "
        "Trainline results or financial impact.",
        "",
        "## Experiment Decision",
        "",
        decision,
        "",
        "## Primary Metric",
        "",
        (f"- Control session booking conversion: **{float(p['control_rate']):.2%}**"),
        (f"- Variant session booking conversion: **{float(p['variant_rate']):.2%}**"),
        (
            f"- Absolute effect: **{float(p['absolute_effect']):.2%}** "
            f"(95% CI {float(p['ci_low']):.2%} to "
            f"{float(p['ci_high']):.2%})"
        ),
        f"- Two-sided p-value: **{float(p['p_value']):.6f}**",
        "",
        "## Variance Reduction and Robustness",
        "",
        (f"- CUPED-style adjusted effect: **{float(c['adjusted_effect']):.2%}**"),
        (f"- CUPED variance reduction: **{float(c['variance_reduction']):.2%}**"),
        (f"- Partner-clustered adjusted effect: **{float(a['adjusted_effect']):.2%}**"),
        "",
        "## Randomisation Health",
        "",
        (f"- SRM p-value: **{float(s['p_value']):.6f}**; flag = **{bool(s['srm_flag'])}**"),
        "",
        "## Power",
        "",
        (f"- Approximate observed power: **{power['observed_approximate_power']:.2%}**"),
        (f"- Approximate 80% power MDE: **{power['minimum_detectable_absolute_effect']:.2%}**"),
        "",
        "## Guardrails",
        "",
    ]

    for _, row in guardrails.iterrows():
        unit = " ms" if row["metric"] == "search_latency_ms" else "%"
        if unit == "%":
            effect_text = f"{float(row['absolute_effect']):.2%}"
        else:
            effect_text = f"{float(row['absolute_effect']):,.2f}{unit}"
        lines.append(f"- {row['metric']}: effect **{effect_text}**, p={float(row['p_value']):.6f}")

    lines.extend(
        [
            "",
            "## Exploratory Heterogeneity",
            "",
            (
                f"- Segments tested: **{len(segments)}**; significant after "
                f"Benjamini-Hochberg FDR correction: "
                f"**{len(significant_segments)}**."
            ),
            "",
            "## Synthetic Business Translation",
            "",
            (
                f"- Estimated incremental booking sessions in the variant "
                f"sample: "
                f"**{float(i['estimated_incremental_booking_sessions']):,.2f}**."
            ),
            (
                f"- Estimated incremental successful booking value: "
                f"**£{float(i['estimated_incremental_successful_booking_value']):,.2f}**."
            ),
            "",
            "These impact values are synthetic scenario estimates and must not "
            "be presented as actual Trainline financial results.",
            "",
        ]
    )

    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_MD.write_text("\n".join(lines), encoding="utf-8")

    return findings


def validate_outputs(
    srm: pd.DataFrame,
    balance: pd.DataFrame,
    primary: pd.DataFrame,
    cuped: pd.DataFrame,
    adjusted: pd.DataFrame,
    guardrails: pd.DataFrame,
    segments: pd.DataFrame,
) -> dict[str, object]:
    p = primary.iloc[0]
    checks = {
        "experiment_has_control_and_variant": (int(p["control_n"]) > 0 and int(p["variant_n"]) > 0),
        "primary_rates_in_bounds": (
            0 <= float(p["control_rate"]) <= 1 and 0 <= float(p["variant_rate"]) <= 1
        ),
        "primary_ci_ordered": (
            float(p["ci_low"]) <= float(p["absolute_effect"]) <= float(p["ci_high"])
        ),
        "srm_p_value_valid": (0 <= float(srm.iloc[0]["p_value"]) <= 1),
        "balance_table_nonempty": not balance.empty,
        "cuped_result_finite": bool(
            np.isfinite(cuped.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "cluster_adjusted_result_finite": bool(
            np.isfinite(adjusted.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "guardrail_p_values_valid": bool(guardrails["p_value"].between(0, 1).all()),
        "segment_p_values_valid": bool(segments.empty or segments["p_value"].between(0, 1).all()),
    }

    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("Stage 4 validation failed: " + ", ".join(failures))

    return checks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    try:
        tables = load_experiment_tables(con)

        exp = tables["session"].copy()
        searches = tables["searches"].copy()
        bookings = tables["bookings"].copy()
        preperiod = tables["preperiod"].copy()
        partners = tables["partners"].copy()

        srm = sample_ratio_mismatch(exp)
        balance = build_balance_table(exp)
        primary = build_primary_effect(exp)
        cuped, cuped_data = build_cuped_effect(
            exp,
            preperiod,
        )
        adjusted = build_cluster_adjusted_effect(
            cuped_data,
        )
        guardrails = build_guardrails(
            searches,
            bookings,
        )
        segments = build_segment_effects(
            exp,
            partners,
        )
        power = build_power_mde(primary)
        impact = build_business_impact(primary, exp)

        checks = validate_outputs(
            srm,
            balance,
            primary,
            cuped,
            adjusted,
            guardrails,
            segments,
        )

        outputs = {
            "mart_stage4_srm": srm,
            "mart_stage4_balance": balance,
            "mart_stage4_primary_effect": primary,
            "mart_stage4_cuped_effect": cuped,
            "mart_stage4_cluster_adjusted_effect": adjusted,
            "mart_stage4_guardrails": guardrails,
            "mart_stage4_segment_effects": segments,
            "mart_stage4_business_impact": impact,
        }

        for name, frame in outputs.items():
            write_table(con, name, frame)
            frame.to_csv(
                OUT_DIR / f"{name}.csv",
                index=False,
            )

        (OUT_DIR / "power_mde.json").write_text(
            json.dumps(power, indent=2),
            encoding="utf-8",
        )

        findings = write_findings(
            primary,
            cuped,
            adjusted,
            srm,
            guardrails,
            power,
            segments,
            impact,
        )
        write_experiment_card(srm, primary, power)

        validation = {
            "status": "PASS",
            "data_classification": "synthetic portfolio data",
            "checks": checks,
            "outputs": {name: int(len(frame)) for name, frame in outputs.items()},
        }
        (OUT_DIR / "validation_report.json").write_text(
            json.dumps(validation, indent=2),
            encoding="utf-8",
        )

        p = primary.iloc[0]
        c = cuped.iloc[0]
        a = adjusted.iloc[0]
        s = srm.iloc[0]

        print()
        print("=" * 78)
        print("RAILNEXUS - STAGE 4 EXPERIMENTATION")
        print("=" * 78)
        print(f"Control conversion          : {float(p['control_rate']):.2%}")
        print(f"Variant conversion          : {float(p['variant_rate']):.2%}")
        print(f"Raw absolute effect         : {float(p['absolute_effect']):.2%}")
        print(
            f"Raw 95% CI                  : [{float(p['ci_low']):.2%}, {float(p['ci_high']):.2%}]"
        )
        print(f"Raw p-value                 : {float(p['p_value']):.6f}")
        print(f"CUPED-style effect          : {float(c['adjusted_effect']):.2%}")
        print(f"Cluster-adjusted effect     : {float(a['adjusted_effect']):.2%}")
        print(f"SRM p-value                 : {float(s['p_value']):.6f}")
        print(f"80% power MDE               : {power['minimum_detectable_absolute_effect']:.2%}")
        print(f"Guardrails analysed         : {len(guardrails)}")
        print(f"Segments tested             : {len(segments)}")
        print(f"Stage 4 findings            : {FINDINGS_MD.relative_to(ROOT)}")
        print(f"Experiment card             : {EXPERIMENT_CARD.relative_to(ROOT)}")
        print("STAGE 4                     : PASS")
        print("=" * 78)

        _ = findings

    finally:
        con.close()


if __name__ == "__main__":
    main()
