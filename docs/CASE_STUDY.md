# RailNexus Case Study

## Problem

A B2B rail platform needs a coherent way to understand partner activation,
traveller product friction, experiment impact, network opportunity and future
booking propensity while keeping analytics reproducible and decision-making
governed.

## Approach

RailNexus models the full analytical workflow:

1. deterministic synthetic B2B rail data generation;
2. DuckDB warehouse and product metric marts;
3. partner activation, funnel, retention and growth analysis;
4. rigorous session-level product experimentation;
5. graph/geospatial opportunity analysis;
6. predictive modelling and clustering;
7. AI-assisted anomaly investigation and governance;
8. Streamlit product application;
9. FastAPI, Docker and CI/CD production hardening.

## Product analytics

The synthetic system contains **10,000 sessions**,
**13,602 searches** and
**2,867 bookings**. Overall session booking conversion
is **27.01%**.

The highest-ranked current partner opportunity is
**PTR_0476**, classified as
**High-volume conversion opportunity**. The opportunity engine combines traffic,
commercial value, conversion friction, no-result exposure and recent movement;
it is a prioritisation heuristic rather than a causal score.

## Experimentation

Smart Journey Alternatives is evaluated with session-level intent-to-treat
analysis.

- Control conversion: **27.38%**
- Variant conversion: **26.78%**
- Absolute effect: **-0.60%**
- 95% CI: **[-3.99%, 2.78%]**
- p-value: **0.727150**
- SRM p-value: **0.251299**

The project also includes CUPED-style pre-period adjustment, partner-clustered
inference, power/MDE, guardrails and FDR-corrected exploratory heterogeneity.

## Network intelligence

The graph layer analyses **360 stations** and
**11,822 directed routes** using demand-weighted PageRank,
betweenness centrality and Haversine distance.

The highest-ranked route opportunity in the current synthetic run is
**Paris → Madrid**, classified as
**High no-result demand**.

## Predictive modelling

Candidate models include Logistic Regression, Probit, Random Forest and MLP,
plus a dummy baseline.

The champion is selected using validation PR-AUC. Its operating threshold is
selected from validation predictions and locked before May–June temporal test
evaluation.

- Champion: **MLP**
- Locked test ROC-AUC: **0.5632**
- Locked test PR-AUC: **0.3116**
- Locked test F1: **0.4412**
- Top-decile lift: **1.257x**

The feature contract excludes post-booking outcomes, revenue, experiment
assignment and Smart Alternatives exposure.

## Responsible AI

Daily metrics are monitored against rolling robust baselines. Structured
investigation packets contain observed evidence, segment/partner/route context,
hypotheses to test and recommended owners. The workflow keeps observations
separate from hypotheses, preserves analyst decision ownership and forbids
unsupported causal claims.

## Production engineering

The validated application includes:

- Streamlit multi-page product application;
- FastAPI analytics and booking-propensity endpoints;
- liveness and readiness probes;
- Pydantic contracts;
- request IDs and security headers;
- read-only DuckDB at serving time;
- non-root Docker image and healthcheck;
- hardened Docker Compose settings;
- GitHub Actions CI for deterministic rebuild, lint, tests and Docker build.

Stage 9 production controls report: **PASS**.

## Data notice

Every partner, traveller, route, commercial, experiment and model value in this
repository is synthetic. The project does not claim actual Trainline product or
financial impact.
