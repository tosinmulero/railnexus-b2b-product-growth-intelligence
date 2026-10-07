# RailNexus Experiment Card

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

- Control sessions: 1,293
- Variant sessions: 1,352
- Variant share: 51.12%
- SRM p-value: 0.251299
- SRM flag: False

## Primary effect

- Control conversion: 27.38%
- Variant conversion: 26.78%
- Absolute effect: -0.60%
- 95% CI: [-3.99%, 2.78%]
- p-value: 0.727150

## Power

- Target power: 80%
- Approximate observed power: 6.41%
- Approximate MDE: 4.98%

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
