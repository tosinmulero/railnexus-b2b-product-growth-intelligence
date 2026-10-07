# RailNexus — Stage 5 Network Intelligence Method

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
