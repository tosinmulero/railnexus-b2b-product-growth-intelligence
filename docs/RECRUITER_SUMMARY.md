# RailNexus — Recruiter Summary

## 60-second summary

RailNexus is an end-to-end **B2B rail product intelligence system** built with
synthetic data to demonstrate product analytics, experimentation, machine
learning, graph/geospatial analysis, responsible AI and production engineering.

The project analyses 10,000 sessions,
13,602 searches and 2,867
bookings across 600 active synthetic B2B
partners.

It includes:

- activation, funnel, retention and partner-growth analytics;
- session-level A/B testing with SRM checks, confidence intervals, CUPED-style
  adjustment, clustered inference, power/MDE and guardrails;
- route/station graph analysis with PageRank, betweenness and geospatial
  opportunity scoring;
- Logistic Regression, Probit, Random Forest and MLP candidate models;
- K-Means partner segmentation;
- validation-selected champion model with a validation-locked operating
  threshold;
- AI-assisted anomaly investigation with evidence-grounded governance controls;
- Streamlit product application;
- FastAPI model/analytics service;
- Docker and GitHub Actions CI.

## Current synthetic results

- Session booking conversion: **27.01%**
- Successful booking value: **£234,505**
- Experiment control conversion: **27.38%**
- Experiment variant conversion: **26.78%**
- Experiment absolute effect: **-0.60%**
- Experiment p-value: **0.727150**
- SRM p-value: **0.251299**
- Champion model: **MLP**
- Validation PR-AUC: **0.3232**
- Locked test ROC-AUC: **0.5632**
- Locked test PR-AUC: **0.3116**
- Test top-decile lift: **1.257x**
- Partner clusters: **2**
- Network stations: **360**
- Network routes: **11,822**

All results above are **synthetic portfolio results**, not Trainline results.
