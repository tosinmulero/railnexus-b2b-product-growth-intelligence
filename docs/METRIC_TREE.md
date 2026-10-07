# RailNexus Product Metric Tree

## Business objective

Increase sustainable B2B rail booking value by improving partner activation, traveller experience, partner retention and network coverage.

## North-star metric

**Successful Booking Value per Active Partner**

Successful booking value excludes cancelled and refunded bookings.

```text
successful non-cancelled/non-refunded booking value
---------------------------------------------------
distinct active B2B partners
```

This connects partner adoption, traveller conversion and commercial value while avoiding optimisation of traffic that does not produce completed customer value.

## Partner activation

```text
Contract -> API Key -> Integration Test -> First Search -> First Booking
```

Metrics:
- integration completion rate
- first-search rate
- first-booking rate
- days to first booking
- 30-day activation rate

The 30-day activation denominator includes only partners with a complete 30-day observation window, reducing right-censoring bias.

## Traveller product engagement

```text
Search -> Results -> Selection -> Booking
```

Metrics:
- searches
- result-selection rate
- session booking conversion
- search-to-book conversion
- no-result rate
- Smart Alternatives exposure rate

## Experimentation

**Smart Journey Alternatives**

Primary metric:
- session booking conversion

Guardrails:
- search latency
- cancellation rate
- refund rate

Stage 2 reports descriptive differences only. Statistical inference, power/MDE, variance reduction and heterogeneous treatment effects are reserved for Stage 4.

## Product reliability

- no-result rate
- average search latency
- cancellation rate
- refund rate
- data-quality violations

## Network opportunity

Stage 5 will add:
- route demand and conversion
- graph centrality
- underserved-demand scoring
- partner route opportunity
- geospatial market opportunity

## Data notice

All RailNexus partner, traveller, experiment and financial values are synthetic portfolio data. No proprietary Trainline data is used.
