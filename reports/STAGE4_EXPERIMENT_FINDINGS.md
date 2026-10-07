# RailNexus — Stage 4 Experiment Findings

> All values are synthetic portfolio results. They are not actual Trainline results or financial impact.

## Experiment Decision

The synthetic experiment does not provide sufficient evidence for a positive launch decision at the 5% significance level.

## Primary Metric

- Control session booking conversion: **27.38%**
- Variant session booking conversion: **26.78%**
- Absolute effect: **-0.60%** (95% CI -3.99% to 2.78%)
- Two-sided p-value: **0.727150**

## Variance Reduction and Robustness

- CUPED-style adjusted effect: **-0.57%**
- CUPED variance reduction: **0.04%**
- Partner-clustered adjusted effect: **-0.29%**

## Randomisation Health

- SRM p-value: **0.251299**; flag = **False**

## Power

- Approximate observed power: **6.41%**
- Approximate 80% power MDE: **4.98%**

## Guardrails

- no_result_rate: effect **-0.90%**, p=0.266128
- search_latency_ms: effect **11.84 ms**, p=0.163310
- cancellation_rate: effect **-1.67%**, p=0.205608
- refund_rate: effect **-1.90%**, p=0.076381

## Exploratory Heterogeneity

- Segments tested: **15**; significant after Benjamini-Hochberg FDR correction: **0**.

## Synthetic Business Translation

- Estimated incremental booking sessions in the variant sample: **-8.15**.
- Estimated incremental successful booking value: **£-691.77**.

These impact values are synthetic scenario estimates and must not be presented as actual Trainline financial results.
