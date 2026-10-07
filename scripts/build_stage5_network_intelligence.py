from __future__ import annotations

import json
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
OUT_DIR = ROOT / "reports" / "generated" / "stage5"
FINDINGS_MD = ROOT / "reports" / "STAGE5_NETWORK_FINDINGS.md"
METHOD_MD = ROOT / "docs" / "STAGE5_NETWORK_METHOD.md"
MAP_HTML = OUT_DIR / "route_opportunity_map.html"


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    n = pd.to_numeric(numerator, errors="coerce").fillna(0).astype(float)
    d = pd.to_numeric(denominator, errors="coerce").fillna(0).astype(float)
    return pd.Series(
        np.divide(
            n,
            d,
            out=np.zeros(len(n), dtype=float),
            where=d.to_numpy() != 0,
        ),
        index=n.index,
    )


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    std = float(s.std(ddof=0))
    if std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - float(s.mean())) / std


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_km = 6371.0088
    p1 = radians(lat1)
    p2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlambda / 2) ** 2
    return 2 * radius_km * asin(sqrt(a))


def load_inputs(
    con: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stations = con.execute(
        """
        SELECT
            station_id,
            station_name,
            city,
            country,
            latitude,
            longitude,
            hub_weight
        FROM stg_stations
        """
    ).fetch_df()

    routes = con.execute(
        """
        SELECT *
        FROM mart_route_performance
        """
    ).fetch_df()

    searches = con.execute(
        """
        SELECT
            partner_id,
            origin_station_id,
            destination_station_id,
            booked_flag,
            booking_count,
            successful_booking_value,
            platform_revenue
        FROM fct_search_journey
        """
    ).fetch_df()

    if stations.empty or routes.empty or searches.empty:
        raise RuntimeError("Stage 5 requires non-empty station, route and search tables.")

    return stations, routes, searches


def enrich_routes(
    routes: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    origin = stations.rename(
        columns={
            "station_id": "origin_station_id",
            "station_name": "origin_station_name",
            "city": "origin_city",
            "country": "origin_country",
            "latitude": "origin_latitude",
            "longitude": "origin_longitude",
            "hub_weight": "origin_hub_weight",
        }
    )
    destination = stations.rename(
        columns={
            "station_id": "destination_station_id",
            "station_name": "destination_station_name",
            "city": "destination_city",
            "country": "destination_country",
            "latitude": "destination_latitude",
            "longitude": "destination_longitude",
            "hub_weight": "destination_hub_weight",
        }
    )

    out = routes.merge(
        origin,
        on="origin_station_id",
        how="left",
        validate="many_to_one",
    ).merge(
        destination,
        on="destination_station_id",
        how="left",
        validate="many_to_one",
    )

    required = [
        "origin_latitude",
        "origin_longitude",
        "destination_latitude",
        "destination_longitude",
    ]
    if out[required].isna().any().any():
        raise RuntimeError("Route enrichment produced missing station coordinates.")

    out["distance_km"] = [
        haversine_km(lat1, lon1, lat2, lon2)
        for lat1, lon1, lat2, lon2 in zip(
            out["origin_latitude"],
            out["origin_longitude"],
            out["destination_latitude"],
            out["destination_longitude"],
            strict=True,
        )
    ]
    out["international_flag"] = (out["origin_country"] != out["destination_country"]).astype(int)

    return out


def build_graph(
    enriched_routes: pd.DataFrame,
    stations: pd.DataFrame,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    for row in stations.itertuples(index=False):
        graph.add_node(
            row.station_id,
            station_name=row.station_name,
            city=row.city,
            country=row.country,
            latitude=float(row.latitude),
            longitude=float(row.longitude),
        )

    for row in enriched_routes.itertuples(index=False):
        demand = max(float(row.searches), 1.0)
        conversion = max(float(row.search_to_book_conversion), 1e-6)
        graph.add_edge(
            row.origin_station_id,
            row.destination_station_id,
            searches=float(row.searches),
            bookings=float(row.bookings),
            conversion=float(row.search_to_book_conversion),
            no_result_rate=float(row.no_result_rate),
            successful_booking_value=float(getattr(row, "successful_booking_value", 0.0)),
            distance_km=float(row.distance_km),
            inverse_demand_cost=1.0 / demand,
            conversion_cost=1.0 / conversion,
        )

    return graph


def build_station_network_metrics(
    graph: nx.DiGraph,
    stations: pd.DataFrame,
    enriched_routes: pd.DataFrame,
) -> pd.DataFrame:
    weighted_pagerank = nx.pagerank(
        graph,
        weight="searches",
    )
    in_strength = dict(graph.in_degree(weight="searches"))
    out_strength = dict(graph.out_degree(weight="searches"))

    undirected = graph.to_undirected()
    betweenness = nx.betweenness_centrality(
        undirected,
        weight="inverse_demand_cost",
        normalized=True,
    )

    degree = dict(undirected.degree())

    metrics = stations.copy()
    metrics["pagerank_search_demand"] = metrics["station_id"].map(weighted_pagerank).fillna(0.0)
    metrics["betweenness_centrality"] = metrics["station_id"].map(betweenness).fillna(0.0)
    metrics["network_degree"] = metrics["station_id"].map(degree).fillna(0)
    metrics["inbound_search_strength"] = metrics["station_id"].map(in_strength).fillna(0.0)
    metrics["outbound_search_strength"] = metrics["station_id"].map(out_strength).fillna(0.0)

    origin_rollup = (
        enriched_routes.groupby("origin_station_id", as_index=False)
        .agg(
            outbound_route_searches=("searches", "sum"),
            outbound_bookings=("bookings", "sum"),
            outbound_no_result_searches=("zero_result_searches", "sum"),
            outbound_successful_booking_value=(
                "successful_booking_value",
                "sum",
            ),
        )
        .rename(columns={"origin_station_id": "station_id"})
    )

    metrics = metrics.merge(
        origin_rollup,
        on="station_id",
        how="left",
        validate="one_to_one",
    )
    numeric = [
        "outbound_route_searches",
        "outbound_bookings",
        "outbound_no_result_searches",
        "outbound_successful_booking_value",
    ]
    metrics[numeric] = metrics[numeric].fillna(0)

    metrics["outbound_conversion"] = safe_div(
        metrics["outbound_bookings"],
        metrics["outbound_route_searches"],
    )
    metrics["outbound_no_result_rate"] = safe_div(
        metrics["outbound_no_result_searches"],
        metrics["outbound_route_searches"],
    )

    metrics["station_opportunity_score"] = (
        0.30 * zscore(np.log1p(metrics["outbound_route_searches"]))
        + 0.20 * zscore(metrics["pagerank_search_demand"])
        + 0.20 * zscore(metrics["betweenness_centrality"])
        + 0.18 * zscore(metrics["outbound_no_result_rate"])
        + 0.12
        * zscore(
            (metrics["outbound_conversion"].median() - metrics["outbound_conversion"]).clip(lower=0)
        )
    )

    metrics["station_opportunity_rank"] = (
        metrics["station_opportunity_score"].rank(method="first", ascending=False).astype(int)
    )

    return metrics.sort_values(["station_opportunity_rank", "station_id"]).reset_index(drop=True)


def build_route_opportunities(
    enriched_routes: pd.DataFrame,
    station_metrics: pd.DataFrame,
) -> pd.DataFrame:
    out = enriched_routes.copy()

    centrality = station_metrics[
        [
            "station_id",
            "pagerank_search_demand",
            "betweenness_centrality",
        ]
    ]

    origin = centrality.rename(
        columns={
            "station_id": "origin_station_id",
            "pagerank_search_demand": "origin_pagerank",
            "betweenness_centrality": "origin_betweenness",
        }
    )
    destination = centrality.rename(
        columns={
            "station_id": "destination_station_id",
            "pagerank_search_demand": "destination_pagerank",
            "betweenness_centrality": "destination_betweenness",
        }
    )

    out = out.merge(
        origin,
        on="origin_station_id",
        how="left",
        validate="many_to_one",
    ).merge(
        destination,
        on="destination_station_id",
        how="left",
        validate="many_to_one",
    )

    conversion_gap = (
        float(out["search_to_book_conversion"].median()) - out["search_to_book_conversion"]
    ).clip(lower=0)

    no_result_excess = (out["no_result_rate"] - float(out["no_result_rate"].median())).clip(lower=0)

    endpoint_centrality = (
        out["origin_pagerank"]
        + out["destination_pagerank"]
        + out["origin_betweenness"]
        + out["destination_betweenness"]
    )

    out["underserved_opportunity_score"] = (
        0.32 * zscore(np.log1p(out["searches"]))
        + 0.23 * zscore(conversion_gap)
        + 0.20 * zscore(no_result_excess)
        + 0.15 * zscore(endpoint_centrality)
        + 0.10 * zscore(np.log1p(out["distance_km"]))
    )

    out["opportunity_rank"] = (
        out["underserved_opportunity_score"].rank(method="first", ascending=False).astype(int)
    )

    out["opportunity_type"] = np.select(
        [
            (
                (out["searches"] >= out["searches"].quantile(0.75))
                & (out["search_to_book_conversion"] < out["search_to_book_conversion"].median())
            ),
            out["no_result_rate"] >= out["no_result_rate"].quantile(0.75),
            ((out["international_flag"] == 1) & (out["searches"] >= out["searches"].median())),
        ],
        [
            "High-demand conversion gap",
            "High no-result demand",
            "Cross-border expansion opportunity",
        ],
        default="Network optimisation opportunity",
    )

    return out.sort_values(
        ["opportunity_rank", "origin_station_id", "destination_station_id"]
    ).reset_index(drop=True)


def build_city_market_opportunities(
    route_opportunities: pd.DataFrame,
) -> pd.DataFrame:
    city = route_opportunities.groupby(
        ["origin_city", "origin_country"],
        as_index=False,
    ).agg(
        searches=("searches", "sum"),
        bookings=("bookings", "sum"),
        successful_booking_value=("successful_booking_value", "sum"),
        average_no_result_rate=("no_result_rate", "mean"),
        average_route_score=("underserved_opportunity_score", "mean"),
        high_priority_routes=(
            "opportunity_rank",
            lambda x: int((x <= 100).sum()),
        ),
    )

    city["conversion"] = safe_div(
        city["bookings"],
        city["searches"],
    )
    city["market_opportunity_score"] = (
        0.35 * zscore(np.log1p(city["searches"]))
        + 0.25 * zscore(city["average_no_result_rate"])
        + 0.20 * zscore((city["conversion"].median() - city["conversion"]).clip(lower=0))
        + 0.20 * zscore(city["average_route_score"])
    )

    city["market_rank"] = (
        city["market_opportunity_score"].rank(method="first", ascending=False).astype(int)
    )

    return city.sort_values(["market_rank", "origin_city"]).reset_index(drop=True)


def build_partner_route_opportunities(
    searches: pd.DataFrame,
    route_opportunities: pd.DataFrame,
) -> pd.DataFrame:
    top_routes = route_opportunities.loc[
        route_opportunities["opportunity_rank"] <= 250,
        [
            "origin_station_id",
            "destination_station_id",
            "opportunity_rank",
            "underserved_opportunity_score",
            "opportunity_type",
        ],
    ]

    partner_route = searches.groupby(
        [
            "partner_id",
            "origin_station_id",
            "destination_station_id",
        ],
        as_index=False,
    ).agg(
        searches=("booked_flag", "size"),
        booked_searches=("booked_flag", "sum"),
        bookings=("booking_count", "sum"),
        successful_booking_value=("successful_booking_value", "sum"),
    )

    partner_route["conversion"] = safe_div(
        partner_route["booked_searches"],
        partner_route["searches"],
    )

    out = partner_route.merge(
        top_routes,
        on=["origin_station_id", "destination_station_id"],
        how="inner",
        validate="many_to_one",
    )

    if out.empty:
        return out

    out["partner_route_opportunity_score"] = (
        0.45 * zscore(out["underserved_opportunity_score"])
        + 0.30 * zscore(np.log1p(out["searches"]))
        + 0.25 * zscore((out["conversion"].median() - out["conversion"]).clip(lower=0))
    )

    out["partner_route_rank"] = (
        out["partner_route_opportunity_score"].rank(method="first", ascending=False).astype(int)
    )

    return out.sort_values(["partner_route_rank", "partner_id"]).reset_index(drop=True)


def build_map(
    route_opportunities: pd.DataFrame,
    station_metrics: pd.DataFrame,
) -> None:
    top_routes = route_opportunities.head(75).copy()
    top_stations = station_metrics.head(50).copy()

    fig = go.Figure()

    for row in top_routes.itertuples(index=False):
        fig.add_trace(
            go.Scattergeo(
                lon=[
                    row.origin_longitude,
                    row.destination_longitude,
                ],
                lat=[
                    row.origin_latitude,
                    row.destination_latitude,
                ],
                mode="lines",
                line={"width": 1},
                opacity=0.30,
                hoverinfo="text",
                text=(
                    f"{row.origin_city} → {row.destination_city}<br>"
                    f"Searches: {int(row.searches):,}<br>"
                    f"Conversion: {row.search_to_book_conversion:.2%}<br>"
                    f"No-result: {row.no_result_rate:.2%}<br>"
                    f"Opportunity rank: {int(row.opportunity_rank)}"
                ),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scattergeo(
            lon=top_stations["longitude"],
            lat=top_stations["latitude"],
            mode="markers",
            marker={
                "size": np.clip(
                    7
                    + 4
                    * (
                        top_stations["station_opportunity_score"]
                        - top_stations["station_opportunity_score"].min()
                    ),
                    7,
                    22,
                ),
            },
            text=[
                (
                    f"{name}<br>{city}, {country}<br>"
                    f"Station opportunity rank: {rank}<br>"
                    f"PageRank: {pagerank:.5f}<br>"
                    f"Betweenness: {betweenness:.5f}"
                )
                for name, city, country, rank, pagerank, betweenness in zip(
                    top_stations["station_name"],
                    top_stations["city"],
                    top_stations["country"],
                    top_stations["station_opportunity_rank"],
                    top_stations["pagerank_search_demand"],
                    top_stations["betweenness_centrality"],
                    strict=True,
                )
            ],
            hoverinfo="text",
            name="Priority stations",
        )
    )

    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        fitbounds="locations",
    )
    fig.update_layout(
        title=("RailNexus — Synthetic B2B Rail Network Opportunity Map"),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    MAP_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        MAP_HTML,
        include_plotlyjs="cdn",
        full_html=True,
    )


def write_table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    frame: pd.DataFrame,
) -> None:
    temp = f"_stage5_{name}"
    con.register(temp, frame)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "{temp}"')
    finally:
        con.unregister(temp)


def write_methodology() -> None:
    text = """# RailNexus — Stage 5 Network Intelligence Method

## Data classification

All station, route, partner, demand and commercial values are synthetic
portfolio data. No proprietary Trainline data is used.

## Network model

The rail network is represented as a directed graph:

- nodes = synthetic stations;
- directed edges = searched origin-destination routes;
- edge demand weight = search volume;
- edge conversion = booked-search rate;
- edge friction = no-result rate;
- edge commercial value = successful booking value.

## Graph metrics

### Weighted PageRank

Search-volume-weighted PageRank estimates how structurally important a station
is within the observed demand network.

### Betweenness centrality

Betweenness centrality is calculated on an undirected projection using inverse
search demand as the path cost. High values highlight stations that bridge
important demand corridors.

### Inbound and outbound search strength

Weighted in-degree and out-degree quantify directional search demand.

## Geospatial features

Great-circle distance is computed with the Haversine formula from station
latitude and longitude.

Routes are also classified as domestic or international.

## Underserved route opportunity score

The composite route score combines:

1. search demand;
2. conversion gap;
3. excess no-result rate;
4. endpoint network centrality;
5. route distance.

The score is a portfolio prioritisation heuristic, not a causal estimate.

## Market and partner opportunity layers

Route scores are aggregated to origin-city market opportunities and joined to
partner route demand to surface B2B expansion candidates.

## Interpretation

These outputs are intended to demonstrate product-market-fit, geospatial and
graph-analysis capability. They must not be presented as real Trainline route,
station, partner or financial performance.
"""
    METHOD_MD.parent.mkdir(parents=True, exist_ok=True)
    METHOD_MD.write_text(text, encoding="utf-8")


def write_findings(
    station_metrics: pd.DataFrame,
    route_opportunities: pd.DataFrame,
    city_opportunities: pd.DataFrame,
    partner_route: pd.DataFrame,
) -> dict[str, object]:
    top_station = station_metrics.iloc[0]
    top_route = route_opportunities.iloc[0]
    top_city = city_opportunities.iloc[0]

    findings: dict[str, object] = {
        "data_classification": "synthetic portfolio data",
        "top_station_opportunity": {
            "station_id": str(top_station["station_id"]),
            "station_name": str(top_station["station_name"]),
            "city": str(top_station["city"]),
            "country": str(top_station["country"]),
            "rank": int(top_station["station_opportunity_rank"]),
            "pagerank": float(top_station["pagerank_search_demand"]),
            "betweenness": float(top_station["betweenness_centrality"]),
        },
        "top_route_opportunity": {
            "origin_station_id": str(top_route["origin_station_id"]),
            "destination_station_id": str(top_route["destination_station_id"]),
            "origin_city": str(top_route["origin_city"]),
            "destination_city": str(top_route["destination_city"]),
            "searches": int(top_route["searches"]),
            "conversion": float(top_route["search_to_book_conversion"]),
            "no_result_rate": float(top_route["no_result_rate"]),
            "opportunity_type": str(top_route["opportunity_type"]),
            "rank": int(top_route["opportunity_rank"]),
        },
        "top_city_market": {
            "city": str(top_city["origin_city"]),
            "country": str(top_city["origin_country"]),
            "searches": int(top_city["searches"]),
            "conversion": float(top_city["conversion"]),
            "market_rank": int(top_city["market_rank"]),
        },
        "network_scale": {
            "stations": int(len(station_metrics)),
            "routes": int(len(route_opportunities)),
            "cities": int(len(city_opportunities)),
        },
    }

    if not partner_route.empty:
        top_partner = partner_route.iloc[0]
        findings["top_partner_route_opportunity"] = {
            "partner_id": str(top_partner["partner_id"]),
            "origin_station_id": str(top_partner["origin_station_id"]),
            "destination_station_id": str(top_partner["destination_station_id"]),
            "searches": int(top_partner["searches"]),
            "conversion": float(top_partner["conversion"]),
            "rank": int(top_partner["partner_route_rank"]),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stage5_findings.json").write_text(
        json.dumps(findings, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# RailNexus — Stage 5 Network Intelligence Findings",
        "",
        "> All route, station, partner and commercial values are synthetic portfolio data.",
        "",
        "## Network Scale",
        "",
        (
            f"- Analysed **{len(station_metrics):,} stations** and "
            f"**{len(route_opportunities):,} observed directed routes**."
        ),
        "",
        "## Highest-Priority Station",
        "",
        (
            f"- **{top_station['station_name']}** in "
            f"**{top_station['city']}, {top_station['country']}** ranks first "
            "on the composite station opportunity score."
        ),
        (
            f"- Weighted PageRank: "
            f"**{float(top_station['pagerank_search_demand']):.5f}**; "
            f"betweenness centrality: "
            f"**{float(top_station['betweenness_centrality']):.5f}**."
        ),
        "",
        "## Highest-Priority Route",
        "",
        (
            f"- **{top_route['origin_city']} → "
            f"{top_route['destination_city']}** ranks first for network "
            f"opportunity and is classified as "
            f"**{top_route['opportunity_type']}**."
        ),
        (
            f"- Searches: **{int(top_route['searches']):,}**; "
            f"search-to-book conversion: "
            f"**{float(top_route['search_to_book_conversion']):.2%}**; "
            f"no-result rate: **{float(top_route['no_result_rate']):.2%}**."
        ),
        "",
        "## Highest-Priority Origin Market",
        "",
        (
            f"- **{top_city['origin_city']}, "
            f"{top_city['origin_country']}** ranks first on the origin-market "
            "opportunity index."
        ),
        "",
    ]

    if not partner_route.empty:
        top_partner = partner_route.iloc[0]
        lines.extend(
            [
                "## B2B Partner Route Opportunity",
                "",
                (
                    f"- **{top_partner['partner_id']}** is the highest-ranked "
                    "partner-route opportunity among the top underserved "
                    "network routes."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Product Interpretation",
            "",
            "- Use the route score to prioritise supply, merchandising or "
            "alternative-journey investigations.",
            "- Use centrality to distinguish local route friction from "
            "network-structural opportunities.",
            "- Use partner-route demand to identify B2B expansion candidates.",
            "- Treat these scores as prioritisation heuristics, not causal effects.",
            "",
            "The interactive HTML map is available in "
            "`reports/generated/stage5/route_opportunity_map.html`.",
            "",
        ]
    )

    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_MD.write_text("\n".join(lines), encoding="utf-8")

    return findings


def validate_outputs(
    station_metrics: pd.DataFrame,
    route_opportunities: pd.DataFrame,
    city_opportunities: pd.DataFrame,
    partner_route: pd.DataFrame,
) -> dict[str, bool]:
    checks = {
        "station_metrics_nonempty": not station_metrics.empty,
        "route_opportunities_nonempty": not route_opportunities.empty,
        "city_opportunities_nonempty": not city_opportunities.empty,
        "station_rank_unique": (station_metrics["station_opportunity_rank"].is_unique),
        "route_rank_unique": (route_opportunities["opportunity_rank"].is_unique),
        "city_rank_unique": city_opportunities["market_rank"].is_unique,
        "route_distances_positive": bool((route_opportunities["distance_km"] > 0).all()),
        "route_conversion_valid": bool(
            route_opportunities["search_to_book_conversion"].between(0, 1).all()
        ),
        "route_no_result_valid": bool(route_opportunities["no_result_rate"].between(0, 1).all()),
        "station_scores_finite": bool(
            np.isfinite(station_metrics["station_opportunity_score"]).all()
        ),
        "route_scores_finite": bool(
            np.isfinite(route_opportunities["underserved_opportunity_score"]).all()
        ),
        "partner_route_rank_valid": bool(
            partner_route.empty or partner_route["partner_route_rank"].is_unique
        ),
    }

    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("Stage 5 validation failed: " + ", ".join(failures))

    return checks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    try:
        stations, routes, searches = load_inputs(con)
        enriched_routes = enrich_routes(routes, stations)
        graph = build_graph(enriched_routes, stations)
        station_metrics = build_station_network_metrics(
            graph,
            stations,
            enriched_routes,
        )
        route_opportunities = build_route_opportunities(
            enriched_routes,
            station_metrics,
        )
        city_opportunities = build_city_market_opportunities(route_opportunities)
        partner_route = build_partner_route_opportunities(
            searches,
            route_opportunities,
        )

        checks = validate_outputs(
            station_metrics,
            route_opportunities,
            city_opportunities,
            partner_route,
        )

        outputs = {
            "mart_stage5_station_network": station_metrics,
            "mart_stage5_route_opportunities": route_opportunities,
            "mart_stage5_city_market_opportunities": city_opportunities,
            "mart_stage5_partner_route_opportunities": partner_route,
        }

        for name, frame in outputs.items():
            write_table(con, name, frame)
            frame.to_csv(
                OUT_DIR / f"{name}.csv",
                index=False,
            )

        build_map(
            route_opportunities,
            station_metrics,
        )
        write_methodology()
        findings = write_findings(
            station_metrics,
            route_opportunities,
            city_opportunities,
            partner_route,
        )

        validation = {
            "status": "PASS",
            "data_classification": "synthetic portfolio data",
            "graph": {
                "nodes": int(graph.number_of_nodes()),
                "directed_edges": int(graph.number_of_edges()),
            },
            "checks": checks,
            "outputs": {name: int(len(frame)) for name, frame in outputs.items()},
        }
        (OUT_DIR / "validation_report.json").write_text(
            json.dumps(validation, indent=2),
            encoding="utf-8",
        )

        print()
        print("=" * 78)
        print("RAILNEXUS - STAGE 5 NETWORK INTELLIGENCE")
        print("=" * 78)
        print(f"Graph nodes                  : {graph.number_of_nodes():,}")
        print(f"Directed graph edges         : {graph.number_of_edges():,}")
        print(
            f"Top station opportunity      : {findings['top_station_opportunity']['station_name']}"
        )
        print(
            f"Top route opportunity        : "
            f"{findings['top_route_opportunity']['origin_city']} -> "
            f"{findings['top_route_opportunity']['destination_city']}"
        )
        print(f"Top origin market            : {findings['top_city_market']['city']}")
        print(f"Interactive map              : {MAP_HTML.relative_to(ROOT)}")
        print(f"Executive findings           : {FINDINGS_MD.relative_to(ROOT)}")
        print("STAGE 5                     : PASS")
        print("=" * 78)

    finally:
        con.close()


if __name__ == "__main__":
    main()
