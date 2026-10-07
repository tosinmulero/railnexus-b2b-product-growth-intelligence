# RailNexus â€” Stage 6 Predictive Modelling Findings

> All metrics are calculated from synthetic portfolio data.

## Champion model

**MLP** was selected using validation PR-AUC before the test set
was evaluated.

## Locked test result

- ROC-AUC: **0.5632**
- PR-AUC: **0.3116**
- F1: **0.4412**
- Top-decile lift: **1.257x**
- Top-decile capture: **12.58%**

## Methodological control

The operating threshold came from the April validation window, not from the
Mayâ€“June test set. The test set therefore remains a locked temporal holdout.

## Parametric benchmark

A Probit model was fitted alongside Logistic Regression to demonstrate
parametric probability modelling and coefficient-level inference.

## Non-parametric models

Random Forest and MLP provide non-linear alternatives to the parametric
benchmarks.

## Partner segmentation

K-Means selected **2 clusters** using silhouette
score. Segment labels are descriptive product-growth labels applied after
clustering.

## Next stage

Stage 7 will add AI-assisted recurring investigation and anomaly triage over
the product, partner and network analytical layers.
