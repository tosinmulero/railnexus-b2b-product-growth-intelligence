# RailNexus â€” Stage 6 Model Card

## Data classification

All partner, traveller, product and commercial values are synthetic portfolio
data. No proprietary Trainline data is used.

## Prediction task

Predict whether a session will produce at least one booking.

## Prediction point

The model is scored after the first search results are available and before a
booking outcome occurs.

## Target

`booked_flag`

## Leakage controls

The model excludes:

- journey selection outcome;
- booking count;
- booking value;
- platform revenue;
- cancellation/refund outcomes;
- experiment assignment;
- Smart Alternatives exposure;
- any post-booking information.

The feature set uses session context, partner attributes and first-search
information available before the booking outcome.

## Temporal evaluation design

- Train: through 31 March 2026
- Validation: 1â€“30 April 2026
- Test: 1 Mayâ€“30 June 2026

Model selection uses validation PR-AUC only.

The operating threshold is selected from validation predictions only and is
then locked before test evaluation.

## Candidate models

- Logistic Regression
- Probit regression
- Random Forest
- Multi-Layer Perceptron
- Dummy prior baseline

## Champion

**MLP**

Validation PR-AUC: **0.3232**

Locked validation threshold: **0.1680**

## Locked test performance

- ROC-AUC: **0.5632**
- PR-AUC: **0.3116**
- Log loss: **0.5811**
- Brier score: **0.1969**
- F1 at locked threshold: **0.4412**
- Precision: **0.2865**
- Recall: **0.9591**
- Top-decile lift: **1.257x**
- Top-decile positive capture: **12.58%**

## Probit benchmark

Test ROC-AUC: **0.5774**

Test PR-AUC: **0.3185**

Probit is retained as an interpretable parametric benchmark rather than used
to override validation-based champion selection.

## Explainability

Top permutation-importance features on validation:

- is_logged_in
- result_count
- partner_type
- search_latency_ms
- integration_type

## Partner clustering

K-Means selected **k=2** using maximum
silhouette score over candidate cluster counts 2â€“6.

The clustering layer is descriptive segmentation, not causal inference.

## Intended use

- prioritise product investigation;
- support partner/product opportunity triage;
- rank sessions by booking propensity;
- compare parametric and non-parametric modelling approaches.

## Limitations

This is a synthetic portfolio system. Performance must not be interpreted as
real Trainline performance, and the model should not be used for consequential
decisions about real customers or partners.
