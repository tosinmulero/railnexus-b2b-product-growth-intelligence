from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"


@st.cache_data(show_spinner=False)
def query(sql: str) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(sql).fetch_df()


def pct(value: float) -> str:
    return f"{value:.2%}"


def money(value: float) -> str:
    return f"£{value:,.0f}"


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
    st.info(
        "Synthetic portfolio data only — no proprietary Trainline, partner, "
        "traveller or commercial data is used."
    )


def overview_page() -> None:
    page_header(
        "Executive Overview",
        "B2B rail product growth, experimentation, network intelligence and ML.",
    )

    kpi = query(
        """
        SELECT
            COUNT(*) AS sessions,
            SUM(search_count) AS searches,
            SUM(booking_count) AS bookings,
            AVG(booked_flag) AS session_booking_conversion,
            SUM(successful_booking_value) AS successful_booking_value,
            SUM(platform_revenue) AS platform_revenue,
            COUNT(DISTINCT partner_id) AS active_partners
        FROM fct_session_journey
        """
    ).iloc[0]

    cols = st.columns(6)
    cols[0].metric("Sessions", f"{int(kpi['sessions']):,}")
    cols[1].metric("Searches", f"{int(kpi['searches']):,}")
    cols[2].metric("Bookings", f"{int(kpi['bookings']):,}")
    cols[3].metric(
        "Session Conversion",
        pct(float(kpi["session_booking_conversion"])),
    )
    cols[4].metric(
        "Successful Booking Value",
        money(float(kpi["successful_booking_value"])),
    )
    cols[5].metric("Active Partners", f"{int(kpi['active_partners']):,}")

    daily = query(
        """
        SELECT *
        FROM mart_daily_product_metrics
        ORDER BY metric_date
        """
    )
    daily["metric_date"] = pd.to_datetime(daily["metric_date"])

    metric = st.selectbox(
        "Daily KPI",
        [
            "session_booking_conversion",
            "search_to_book_conversion",
            "no_result_rate",
            "avg_search_latency_ms",
            "successful_booking_value_per_active_partner",
        ],
    )

    fig = px.line(
        daily,
        x="metric_date",
        y=metric,
        markers=True,
        title=f"Daily {metric.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product Funnel")
    funnel = query(
        """
        SELECT *
        FROM mart_stage3_funnel
        ORDER BY stage_order
        """
    )
    fig = px.funnel(
        funnel,
        y="stage",
        x="count",
        title="Search → Results → Selection → Booking",
    )
    st.plotly_chart(fig, use_container_width=True)


def partner_growth_page() -> None:
    page_header(
        "Partner Growth & Retention",
        "Activation, partner opportunity ranking and cohort retention.",
    )

    activation = query(
        """
        SELECT *
        FROM mart_stage3_activation_segments
        """
    )
    dimension = st.selectbox(
        "Activation dimension",
        sorted(activation["dimension"].unique()),
    )
    filtered = activation.loc[activation["dimension"] == dimension].copy()
    filtered = filtered.sort_values(
        "activation_30d_rate",
        ascending=False,
    )

    fig = px.bar(
        filtered,
        x="segment_value",
        y="activation_30d_rate",
        title="30-Day Partner Activation Rate",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-Priority Partner Opportunities")
    opp = query(
        """
        SELECT
            priority_rank,
            partner_id,
            partner_type,
            country,
            integration_type,
            contract_tier,
            sessions,
            searches,
            session_booking_conversion,
            no_result_rate,
            conversion_change_pp,
            successful_value_growth_rate,
            opportunity_type,
            opportunity_score
        FROM mart_stage3_partner_opportunities
        ORDER BY priority_rank
        LIMIT 30
        """
    )
    st.dataframe(opp, use_container_width=True, hide_index=True)

    retention = query(
        """
        SELECT *
        FROM mart_stage3_retention_summary
        ORDER BY cohort_month
        """
    )
    if not retention.empty:
        long = retention.melt(
            id_vars=["cohort_month"],
            value_vars=[col for col in retention.columns if col.startswith("month_")],
            var_name="retention_month",
            value_name="retention_rate",
        ).dropna()

        fig = px.line(
            long,
            x="retention_month",
            y="retention_rate",
            color="cohort_month",
            markers=True,
            title="Partner Retention Cohorts",
        )
        st.plotly_chart(fig, use_container_width=True)


def experimentation_page() -> None:
    page_header(
        "Experimentation",
        "Smart Journey Alternatives — rigorous session-level A/B analysis.",
    )

    primary = query(
        """
        SELECT *
        FROM mart_stage4_primary_effect
        """
    ).iloc[0]

    cols = st.columns(5)
    cols[0].metric("Control", pct(float(primary["control_rate"])))
    cols[1].metric("Variant", pct(float(primary["variant_rate"])))
    cols[2].metric(
        "Absolute Effect",
        pct(float(primary["absolute_effect"])),
    )
    cols[3].metric("p-value", f"{float(primary['p_value']):.4g}")
    cols[4].metric(
        "Relative Effect",
        pct(float(primary["relative_effect"])),
    )

    effects = pd.DataFrame(
        {
            "arm": ["Control", "Variant"],
            "conversion": [
                float(primary["control_rate"]),
                float(primary["variant_rate"]),
            ],
        }
    )
    fig = px.bar(
        effects,
        x="arm",
        y="conversion",
        title="Session Booking Conversion",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Guardrails")
    guardrails = query(
        """
        SELECT *
        FROM mart_stage4_guardrails
        ORDER BY metric
        """
    )
    st.dataframe(guardrails, use_container_width=True, hide_index=True)

    st.subheader("Methodological Checks")
    srm = query("SELECT * FROM mart_stage4_srm")
    cuped = query("SELECT * FROM mart_stage4_cuped_effect")
    adjusted = query("SELECT * FROM mart_stage4_cluster_adjusted_effect")

    c1, c2, c3 = st.columns(3)
    c1.metric("SRM p-value", f"{float(srm.iloc[0]['p_value']):.4g}")
    c2.metric(
        "CUPED Adjusted Effect",
        pct(float(cuped.iloc[0]["adjusted_effect"])),
    )
    c3.metric(
        "Cluster-Adjusted Effect",
        pct(float(adjusted.iloc[0]["adjusted_effect"])),
    )


def network_page() -> None:
    page_header(
        "Geospatial & Network Intelligence",
        "Graph centrality, route opportunity and B2B expansion prioritisation.",
    )

    routes = query(
        """
        SELECT
            opportunity_rank,
            origin_city,
            destination_city,
            origin_country,
            destination_country,
            searches,
            bookings,
            search_to_book_conversion,
            no_result_rate,
            distance_km,
            opportunity_type,
            underserved_opportunity_score
        FROM mart_stage5_route_opportunities
        ORDER BY opportunity_rank
        LIMIT 50
        """
    )

    fig = px.scatter(
        routes,
        x="search_to_book_conversion",
        y="no_result_rate",
        size="searches",
        hover_name="opportunity_type",
        hover_data=[
            "origin_city",
            "destination_city",
            "distance_km",
            "opportunity_rank",
        ],
        title="Route Opportunity Landscape",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Route Opportunities")
    st.dataframe(routes, use_container_width=True, hide_index=True)

    stations = query(
        """
        SELECT
            station_opportunity_rank,
            station_name,
            city,
            country,
            network_degree,
            pagerank_search_demand,
            betweenness_centrality,
            outbound_conversion,
            outbound_no_result_rate,
            station_opportunity_score
        FROM mart_stage5_station_network
        ORDER BY station_opportunity_rank
        LIMIT 30
        """
    )
    st.subheader("Network-Critical Stations")
    st.dataframe(stations, use_container_width=True, hide_index=True)

    map_path = ROOT / "reports" / "generated" / "stage5" / "route_opportunity_map.html"
    if map_path.exists():
        st.caption("Interactive geospatial route map is also available in the Stage 5 output.")


def predictive_page() -> None:
    page_header(
        "Predictive Modelling & Segmentation",
        "Temporal validation, booking propensity and partner clustering.",
    )

    comparison = query(
        """
        SELECT *
        FROM mart_stage6_model_validation
        ORDER BY pr_auc DESC
        """
    )

    fig = px.bar(
        comparison,
        x="model",
        y="pr_auc",
        hover_data=["roc_auc", "log_loss", "brier", "f1"],
        title="Validation PR-AUC by Candidate Model",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    importance = query(
        """
        SELECT *
        FROM mart_stage6_feature_importance
        ORDER BY importance_rank
        LIMIT 15
        """
    )
    fig = px.bar(
        importance.sort_values(
            "importance_mean_pr_auc",
            ascending=True,
        ),
        x="importance_mean_pr_auc",
        y="feature",
        orientation="h",
        title="Champion Permutation Importance",
    )
    st.plotly_chart(fig, use_container_width=True)

    clusters = query(
        """
        SELECT *
        FROM mart_stage6_cluster_profile
        ORDER BY cluster_id
        """
    )
    st.subheader("Partner Segments")
    st.dataframe(clusters, use_container_width=True, hide_index=True)


def investigation_page() -> None:
    page_header(
        "AI-Assisted Investigation",
        "Evidence-first anomaly triage with mandatory human review.",
    )

    cases = query(
        """
        SELECT *
        FROM mart_stage7_investigation_cases
        ORDER BY
            CASE severity
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                ELSE 3
            END,
            metric_date DESC
        """
    )

    anomalies = query(
        """
        SELECT *
        FROM mart_stage7_daily_anomalies
        ORDER BY ABS(robust_z) DESC
        """
    )

    c1, c2 = st.columns(2)
    c1.metric("Detected Anomalies", f"{len(anomalies):,}")
    c2.metric("Investigation Cases", f"{len(cases):,}")

    if cases.empty:
        st.success(
            "No metric crossed the configured adverse anomaly threshold "
            "for the current synthetic run."
        )
    else:
        st.subheader("Investigation Queue")
        st.dataframe(cases, use_container_width=True, hide_index=True)

        severity = (
            cases.groupby("severity", as_index=False).size().rename(columns={"size": "cases"})
        )
        fig = px.bar(
            severity,
            x="severity",
            y="cases",
            title="Cases by Severity",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Responsible-AI controls**

        - observations are separated from hypotheses;
        - anomaly detection does not make causal claims;
        - no missing evidence is invented;
        - every escalation requires human review;
        - all values remain synthetic portfolio data.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="RailNexus | B2B Rail Product Intelligence",
        page_icon="🚆",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            border: 1px solid rgba(120,120,120,0.20);
            border-radius: 12px;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("RailNexus")
    st.sidebar.caption("B2B Product Growth Intelligence")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Overview",
            "Partner Growth",
            "Experimentation",
            "Network Intelligence",
            "Predictive Modelling",
            "AI Investigation",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Synthetic portfolio system demonstrating product analytics, "
        "experimentation, graph/geospatial analysis, ML and responsible AI."
    )

    pages = {
        "Executive Overview": overview_page,
        "Partner Growth": partner_growth_page,
        "Experimentation": experimentation_page,
        "Network Intelligence": network_page,
        "Predictive Modelling": predictive_page,
        "AI Investigation": investigation_page,
    }

    pages[page]()


if __name__ == "__main__":
    main()
