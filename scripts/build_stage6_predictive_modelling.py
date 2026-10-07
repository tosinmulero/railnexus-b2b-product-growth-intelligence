from __future__ import annotations

import json
import warnings
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
OUT_DIR = ROOT / "reports" / "generated" / "stage6"
MODEL_DIR = ROOT / "artifacts" / "models"
MODEL_CARD = ROOT / "docs" / "STAGE6_MODEL_CARD.md"
FINDINGS_MD = ROOT / "reports" / "STAGE6_MODEL_FINDINGS.md"

RANDOM_STATE = 20261007


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


TRAIN_END = pd.Timestamp("2026-03-31 23:59:59")
VALIDATION_START = pd.Timestamp("2026-04-01")
VALIDATION_END = pd.Timestamp("2026-04-30 23:59:59")
TEST_START = pd.Timestamp("2026-05-01")

CATEGORICAL_FEATURES = [
    "device_type",
    "acquisition_channel",
    "traveller_type",
    "partner_type",
    "partner_country",
    "integration_type",
    "contract_tier",
    "journey_type",
    "origin_country",
    "destination_country",
    "search_dow",
]

NUMERIC_FEATURES = [
    "is_logged_in",
    "passengers",
    "days_before_travel",
    "result_count",
    "no_result_flag",
    "search_latency_ms",
    "international_flag",
    "search_hour",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_model_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frame = con.execute(
        """
        WITH first_search AS (
            SELECT *
            FROM (
                SELECT
                    f.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY session_id
                        ORDER BY search_ts, search_id
                    ) AS rn
                FROM fct_search_journey f
            )
            WHERE rn = 1
        )
        SELECT
            s.session_id,
            s.partner_id,
            s.session_ts,
            s.booked_flag,
            s.successful_booking_value,
            s.device_type,
            s.acquisition_channel,
            s.is_logged_in,
            s.traveller_type,
            p.partner_type,
            p.country AS partner_country,
            p.integration_type,
            p.contract_tier,
            fs.search_ts,
            fs.journey_type,
            fs.passengers,
            fs.days_before_travel,
            fs.result_count,
            fs.no_result_flag,
            fs.search_latency_ms,
            fs.origin_station_id,
            fs.destination_station_id,
            os.country AS origin_country,
            ds.country AS destination_country
        FROM fct_session_journey s
        INNER JOIN stg_partners p USING (partner_id)
        INNER JOIN first_search fs USING (session_id, partner_id)
        LEFT JOIN stg_stations os
            ON fs.origin_station_id = os.station_id
        LEFT JOIN stg_stations ds
            ON fs.destination_station_id = ds.station_id
        """
    ).fetch_df()

    if frame.empty:
        raise RuntimeError("Stage 6 model frame is empty.")

    frame["session_ts"] = pd.to_datetime(frame["session_ts"])
    frame["search_ts"] = pd.to_datetime(frame["search_ts"])
    frame["search_hour"] = frame["search_ts"].dt.hour.astype(int)
    frame["search_dow"] = frame["search_ts"].dt.day_name()
    frame["international_flag"] = (frame["origin_country"] != frame["destination_country"]).astype(
        int
    )
    frame["is_logged_in"] = frame["is_logged_in"].astype(int)
    frame["booked_flag"] = frame["booked_flag"].astype(int)

    missing = frame[FEATURES].isna().sum()
    if int(missing.sum()) > 0:
        raise RuntimeError(
            "Model frame contains missing feature values: " + str(missing[missing > 0].to_dict())
        )

    return frame


def temporal_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["session_ts"] <= TRAIN_END].copy()
    validation = frame.loc[
        (frame["session_ts"] >= VALIDATION_START) & (frame["session_ts"] <= VALIDATION_END)
    ].copy()
    test = frame.loc[frame["session_ts"] >= TEST_START].copy()

    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if len(split) < 100:
            raise RuntimeError(f"{name} split is too small: {len(split)} rows.")
        if split["booked_flag"].nunique() < 2:
            raise RuntimeError(f"{name} split contains only one target class.")

    return train, validation, test


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Dummy": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=14,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight=None,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.001,
                        batch_size="auto",
                        learning_rate_init=0.001,
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def probability_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def best_f1_threshold(
    y_true: pd.Series,
    probability: np.ndarray,
) -> float:
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probability,
    )
    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx])


def threshold_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def fit_sklearn_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    dict[str, Pipeline],
    pd.DataFrame,
    dict[str, float],
]:
    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}

    x_train = train[FEATURES]
    y_train = train["booked_flag"]
    x_val = validation[FEATURES]
    y_val = validation["booked_flag"]

    for name, pipeline in candidate_models().items():
        print(f"Training {name}...")
        pipeline.fit(x_train, y_train)
        prob = pipeline.predict_proba(x_val)[:, 1]
        threshold = best_f1_threshold(y_val, prob)

        row = {
            "model": name,
            "split": "validation",
            **probability_metrics(y_val, prob),
            **threshold_metrics(y_val, prob, threshold),
        }
        rows.append(row)
        thresholds[name] = threshold
        fitted[name] = pipeline

    comparison = pd.DataFrame(rows)
    return fitted, comparison, thresholds


def fit_probit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[object, pd.DataFrame, dict[str, float], dict[str, float]]:
    formula = (
        "booked_flag ~ "
        "C(device_type) + "
        "C(acquisition_channel) + "
        "C(traveller_type) + "
        "is_logged_in + "
        "C(partner_type) + "
        "C(partner_country) + "
        "C(integration_type) + "
        "C(contract_tier) + "
        "C(journey_type) + "
        "passengers + "
        "days_before_travel + "
        "result_count + "
        "no_result_flag + "
        "search_latency_ms + "
        "C(origin_country) + "
        "C(destination_country) + "
        "international_flag + "
        "search_hour + "
        "C(search_dow)"
    )

    try:
        model = smf.glm(
            formula=formula,
            data=train,
            family=sm.families.Binomial(link=sm.families.links.Probit()),
        ).fit(maxiter=250)
        formula_used = formula
    except Exception:
        fallback = (
            "booked_flag ~ "
            "C(device_type) + "
            "C(acquisition_channel) + "
            "C(traveller_type) + "
            "is_logged_in + "
            "C(partner_type) + "
            "C(integration_type) + "
            "C(contract_tier) + "
            "C(journey_type) + "
            "passengers + "
            "days_before_travel + "
            "result_count + "
            "no_result_flag + "
            "search_latency_ms + "
            "international_flag + "
            "search_hour"
        )
        model = smf.glm(
            formula=fallback,
            data=train,
            family=sm.families.Binomial(link=sm.families.links.Probit()),
        ).fit(maxiter=250)
        formula_used = fallback

    val_prob = np.clip(
        np.asarray(model.predict(validation), dtype=float),
        1e-6,
        1 - 1e-6,
    )
    test_prob = np.clip(
        np.asarray(model.predict(test), dtype=float),
        1e-6,
        1 - 1e-6,
    )

    val_threshold = best_f1_threshold(
        validation["booked_flag"],
        val_prob,
    )
    val_metrics = {
        "model": "Probit",
        "split": "validation",
        **probability_metrics(validation["booked_flag"], val_prob),
        **threshold_metrics(
            validation["booked_flag"],
            val_prob,
            val_threshold,
        ),
    }
    test_metrics = {
        **probability_metrics(test["booked_flag"], test_prob),
        **threshold_metrics(
            test["booked_flag"],
            test_prob,
            val_threshold,
        ),
    }

    ci = model.conf_int()
    coefficients = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "p_value": model.pvalues.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
        }
    )
    coefficients["formula_used"] = formula_used

    return model, coefficients, val_metrics, test_metrics


def select_champion(
    comparison: pd.DataFrame,
) -> str:
    eligible = comparison.loc[
        comparison["model"].isin(["LogisticRegression", "RandomForest", "MLP"])
    ].copy()

    eligible = eligible.sort_values(
        ["pr_auc", "roc_auc", "log_loss"],
        ascending=[False, False, True],
    )
    return str(eligible.iloc[0]["model"])


def evaluate_champion(
    champion_name: str,
    champion: Pipeline,
    threshold: float,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    probability = champion.predict_proba(test[FEATURES])[:, 1]
    metrics = {
        "model": champion_name,
        "threshold_source": "validation",
        "locked_threshold": float(threshold),
        **probability_metrics(test["booked_flag"], probability),
        **threshold_metrics(
            test["booked_flag"],
            probability,
            threshold,
        ),
    }

    scored = test[
        [
            "session_id",
            "partner_id",
            "session_ts",
            "booked_flag",
        ]
    ].copy()
    scored["booking_probability"] = probability
    scored["predicted_booking"] = (probability >= threshold).astype(int)

    scored = scored.sort_values(
        "booking_probability",
        ascending=False,
    ).reset_index(drop=True)

    n_top = max(1, int(np.ceil(len(scored) * 0.10)))
    top = scored.head(n_top)
    overall_rate = float(scored["booked_flag"].mean())
    top_rate = float(top["booked_flag"].mean())
    positives = int(scored["booked_flag"].sum())

    ranking = {
        "overall_booking_rate": overall_rate,
        "top_decile_booking_rate": top_rate,
        "top_decile_lift": (float(top_rate / overall_rate) if overall_rate > 0 else 0.0),
        "top_decile_positive_capture": (
            float(top["booked_flag"].sum() / positives) if positives > 0 else 0.0
        ),
    }

    scored["probability_bin"] = pd.qcut(
        scored["booking_probability"].rank(method="first"),
        q=min(10, len(scored)),
        labels=False,
        duplicates="drop",
    )
    calibration = (
        scored.groupby("probability_bin", as_index=False)
        .agg(
            sessions=("session_id", "count"),
            mean_predicted_probability=(
                "booking_probability",
                "mean",
            ),
            observed_booking_rate=("booked_flag", "mean"),
        )
        .sort_values("probability_bin")
    )

    return scored, calibration, {**metrics, **ranking}


def build_feature_importance(
    champion_name: str,
    champion: Pipeline,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    sample = validation.sample(
        n=min(2500, len(validation)),
        random_state=RANDOM_STATE,
    )

    result = permutation_importance(
        champion,
        sample[FEATURES],
        sample["booked_flag"],
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    out = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean_pr_auc": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    out["model"] = champion_name
    out["importance_rank"] = (
        out["importance_mean_pr_auc"].rank(method="first", ascending=False).astype(int)
    )
    return out.sort_values("importance_rank").reset_index(drop=True)


def model_specific_importance(
    fitted: dict[str, Pipeline],
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    logistic = fitted["LogisticRegression"]
    log_names = logistic.named_steps["preprocess"].get_feature_names_out()
    log_coef = logistic.named_steps["model"].coef_[0]
    log_df = pd.DataFrame(
        {
            "feature": log_names,
            "coefficient": log_coef,
            "absolute_coefficient": np.abs(log_coef),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    )
    outputs["logistic_coefficients"] = log_df.reset_index(drop=True)

    forest = fitted["RandomForest"]
    rf_names = forest.named_steps["preprocess"].get_feature_names_out()
    rf_imp = forest.named_steps["model"].feature_importances_
    rf_df = pd.DataFrame(
        {
            "feature": rf_names,
            "feature_importance": rf_imp,
        }
    ).sort_values(
        "feature_importance",
        ascending=False,
    )
    outputs["random_forest_feature_importance"] = rf_df.reset_index(drop=True)

    return outputs


def load_partner_performance(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            partner_id,
            partner_type,
            country,
            integration_type,
            contract_tier,
            sessions,
            searches,
            bookings,
            session_booking_conversion,
            no_result_rate,
            avg_search_latency_ms,
            successful_booking_value,
            platform_revenue
        FROM mart_partner_performance
        WHERE sessions > 0
        """
    ).fetch_df()


def build_partner_clusters(
    partner: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], object]:
    if len(partner) < 8:
        raise RuntimeError("Not enough active partners for meaningful clustering.")

    x = pd.DataFrame(
        {
            "log_sessions": np.log1p(partner["sessions"]),
            "log_searches": np.log1p(partner["searches"]),
            "session_booking_conversion": partner["session_booking_conversion"].fillna(0),
            "no_result_rate": partner["no_result_rate"].fillna(0),
            "avg_search_latency_ms": partner["avg_search_latency_ms"].fillna(0),
            "log_successful_booking_value": np.log1p(partner["successful_booking_value"]),
            "log_platform_revenue": np.log1p(partner["platform_revenue"]),
        }
    )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    max_k = min(6, len(partner) - 1)
    scores = []
    models = {}
    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            n_init=25,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(x_scaled)
        score = float(silhouette_score(x_scaled, labels))
        scores.append({"k": k, "silhouette_score": score})
        models[k] = model

    score_df = pd.DataFrame(scores)
    best_k = int(
        score_df.sort_values(
            ["silhouette_score", "k"],
            ascending=[False, True],
        ).iloc[0]["k"]
    )
    model = models[best_k]
    labels = model.predict(x_scaled)

    clustered = partner.copy()
    clustered["cluster_id"] = labels.astype(int)

    profile = clustered.groupby("cluster_id", as_index=False).agg(
        partners=("partner_id", "nunique"),
        median_sessions=("sessions", "median"),
        median_searches=("searches", "median"),
        mean_conversion=("session_booking_conversion", "mean"),
        mean_no_result_rate=("no_result_rate", "mean"),
        mean_latency_ms=("avg_search_latency_ms", "mean"),
        median_successful_booking_value=(
            "successful_booking_value",
            "median",
        ),
        median_platform_revenue=("platform_revenue", "median"),
    )

    value_cluster = int(
        profile.sort_values(
            "median_successful_booking_value",
            ascending=False,
        ).iloc[0]["cluster_id"]
    )
    friction_cluster = int(
        profile.sort_values(
            "mean_no_result_rate",
            ascending=False,
        ).iloc[0]["cluster_id"]
    )
    conversion_cluster = int(
        profile.sort_values(
            "mean_conversion",
            ascending=True,
        ).iloc[0]["cluster_id"]
    )

    labels_map: dict[int, str] = {}
    labels_map[value_cluster] = "High-Value Scale"

    if friction_cluster not in labels_map:
        labels_map[friction_cluster] = "Availability Friction"

    if conversion_cluster not in labels_map:
        labels_map[conversion_cluster] = "Conversion Opportunity"

    remaining_names = [
        "Core Growth",
        "Emerging Partners",
        "Established Core",
        "Long-Tail Opportunity",
    ]
    for cluster_id in sorted(profile["cluster_id"].astype(int)):
        if cluster_id not in labels_map:
            labels_map[cluster_id] = remaining_names.pop(0)

    clustered["cluster_label"] = clustered["cluster_id"].map(labels_map)
    profile["cluster_label"] = profile["cluster_id"].map(labels_map)

    metadata = {
        "selected_k": best_k,
        "selection_metric": "silhouette_score",
        "silhouette_scores": scores,
        "features": list(x.columns),
        "cluster_labels": {str(k): v for k, v in labels_map.items()},
    }

    artifact = {
        "scaler": scaler,
        "kmeans": model,
        "features": list(x.columns),
        "metadata": metadata,
    }

    return clustered, profile, metadata, artifact


def write_table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    frame: pd.DataFrame,
) -> None:
    temp = f"_stage6_{name}"
    con.register(temp, frame)
    try:
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "{temp}"')
    finally:
        con.unregister(temp)


def write_documents(
    champion_name: str,
    champion_test: dict[str, float],
    validation_comparison: pd.DataFrame,
    probit_test: dict[str, float],
    cluster_metadata: dict[str, object],
    feature_importance: pd.DataFrame,
) -> None:
    best_val = validation_comparison.loc[validation_comparison["model"] == champion_name].iloc[0]
    top_features = feature_importance.head(5)["feature"].tolist()

    model_card = f"""# RailNexus â€” Stage 6 Model Card

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

**{champion_name}**

Validation PR-AUC: **{float(best_val["pr_auc"]):.4f}**

Locked validation threshold: **{float(champion_test["locked_threshold"]):.4f}**

## Locked test performance

- ROC-AUC: **{float(champion_test["roc_auc"]):.4f}**
- PR-AUC: **{float(champion_test["pr_auc"]):.4f}**
- Log loss: **{float(champion_test["log_loss"]):.4f}**
- Brier score: **{float(champion_test["brier"]):.4f}**
- F1 at locked threshold: **{float(champion_test["f1"]):.4f}**
- Precision: **{float(champion_test["precision"]):.4f}**
- Recall: **{float(champion_test["recall"]):.4f}**
- Top-decile lift: **{float(champion_test["top_decile_lift"]):.3f}x**
- Top-decile positive capture: **{float(champion_test["top_decile_positive_capture"]):.2%}**

## Probit benchmark

Test ROC-AUC: **{float(probit_test["roc_auc"]):.4f}**

Test PR-AUC: **{float(probit_test["pr_auc"]):.4f}**

Probit is retained as an interpretable parametric benchmark rather than used
to override validation-based champion selection.

## Explainability

Top permutation-importance features on validation:

{chr(10).join(f"- {feature}" for feature in top_features)}

## Partner clustering

K-Means selected **k={cluster_metadata["selected_k"]}** using maximum
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
"""

    MODEL_CARD.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CARD.write_text(model_card, encoding="utf-8")

    findings = f"""# RailNexus â€” Stage 6 Predictive Modelling Findings

> All metrics are calculated from synthetic portfolio data.

## Champion model

**{champion_name}** was selected using validation PR-AUC before the test set
was evaluated.

## Locked test result

- ROC-AUC: **{float(champion_test["roc_auc"]):.4f}**
- PR-AUC: **{float(champion_test["pr_auc"]):.4f}**
- F1: **{float(champion_test["f1"]):.4f}**
- Top-decile lift: **{float(champion_test["top_decile_lift"]):.3f}x**
- Top-decile capture: **{float(champion_test["top_decile_positive_capture"]):.2%}**

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

K-Means selected **{cluster_metadata["selected_k"]} clusters** using silhouette
score. Segment labels are descriptive product-growth labels applied after
clustering.

## Next stage

Stage 7 will add AI-assisted recurring investigation and anomaly triage over
the product, partner and network analytical layers.
"""
    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_MD.write_text(findings, encoding="utf-8")


def validate_results(
    comparison: pd.DataFrame,
    champion_test: dict[str, float],
    clusters: pd.DataFrame,
    cluster_profile: pd.DataFrame,
) -> dict[str, bool]:
    checks = {
        "validation_models_present": {
            "Dummy",
            "LogisticRegression",
            "RandomForest",
            "MLP",
            "Probit",
        }.issubset(set(comparison["model"])),
        "champion_test_probabilities_valid": (
            0 <= champion_test["roc_auc"] <= 1
            and 0 <= champion_test["pr_auc"] <= 1
            and 0 <= champion_test["brier"] <= 1
        ),
        "threshold_locked_from_validation": (champion_test["threshold_source"] == "validation"),
        "clusters_nonempty": not clusters.empty,
        "cluster_profile_nonempty": not cluster_profile.empty,
        "cluster_assignments_complete": (clusters["cluster_id"].notna().all()),
    }

    checks = {name: bool(value) for name, value in checks.items()}

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Stage 6 validation failed: " + ", ".join(failed))

    return checks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    try:
        frame = load_model_frame(con)
        train, validation, test = temporal_split(frame)

        fitted, comparison, thresholds = fit_sklearn_candidates(
            train,
            validation,
        )

        (
            _probit_model,
            probit_coefficients,
            probit_val_metrics,
            probit_test_metrics,
        ) = fit_probit(
            train,
            validation,
            test,
        )

        comparison = pd.concat(
            [
                comparison,
                pd.DataFrame([probit_val_metrics]),
            ],
            ignore_index=True,
        )

        champion_name = select_champion(comparison)
        champion = fitted[champion_name]
        locked_threshold = thresholds[champion_name]

        (
            test_scored,
            calibration,
            champion_test,
        ) = evaluate_champion(
            champion_name,
            champion,
            locked_threshold,
            test,
        )

        permutation = build_feature_importance(
            champion_name,
            champion,
            validation,
        )

        specific = model_specific_importance(fitted)

        partner = load_partner_performance(con)
        (
            clusters,
            cluster_profile,
            cluster_metadata,
            cluster_artifact,
        ) = build_partner_clusters(partner)

        checks = validate_results(
            comparison,
            champion_test,
            clusters,
            cluster_profile,
        )

        write_table(
            con,
            "mart_stage6_model_validation",
            comparison,
        )
        write_table(
            con,
            "mart_stage6_test_scores",
            test_scored,
        )
        write_table(
            con,
            "mart_stage6_feature_importance",
            permutation,
        )
        write_table(
            con,
            "mart_stage6_partner_clusters",
            clusters,
        )
        write_table(
            con,
            "mart_stage6_cluster_profile",
            cluster_profile,
        )

        comparison.to_csv(
            OUT_DIR / "model_validation_comparison.csv",
            index=False,
        )
        probit_coefficients.to_csv(
            OUT_DIR / "probit_coefficients.csv",
            index=False,
        )
        test_scored.to_csv(
            OUT_DIR / "champion_test_scores.csv",
            index=False,
        )
        calibration.to_csv(
            OUT_DIR / "champion_calibration.csv",
            index=False,
        )
        permutation.to_csv(
            OUT_DIR / "champion_permutation_importance.csv",
            index=False,
        )
        specific["logistic_coefficients"].to_csv(
            OUT_DIR / "logistic_coefficients.csv",
            index=False,
        )
        specific["random_forest_feature_importance"].to_csv(
            OUT_DIR / "random_forest_feature_importance.csv",
            index=False,
        )
        clusters.to_csv(
            OUT_DIR / "partner_clusters.csv",
            index=False,
        )
        cluster_profile.to_csv(
            OUT_DIR / "partner_cluster_profile.csv",
            index=False,
        )

        (OUT_DIR / "champion_test_metrics.json").write_text(
            json.dumps(champion_test, indent=2, default=json_default),
            encoding="utf-8",
        )
        (OUT_DIR / "probit_test_metrics.json").write_text(
            json.dumps(probit_test_metrics, indent=2, default=json_default),
            encoding="utf-8",
        )
        (OUT_DIR / "cluster_metadata.json").write_text(
            json.dumps(cluster_metadata, indent=2, default=json_default),
            encoding="utf-8",
        )

        joblib.dump(
            champion,
            MODEL_DIR / "stage6_booking_champion.joblib",
        )
        joblib.dump(
            cluster_artifact,
            MODEL_DIR / "stage6_partner_kmeans.joblib",
        )

        metadata = {
            "data_classification": "synthetic portfolio data",
            "champion_model": champion_name,
            "selection_metric": "validation_pr_auc",
            "threshold_source": "validation",
            "locked_threshold": float(locked_threshold),
            "train_end": str(TRAIN_END),
            "validation_start": str(VALIDATION_START),
            "validation_end": str(VALIDATION_END),
            "test_start": str(TEST_START),
            "features": FEATURES,
            "excluded_leakage_features": [
                "selected_flag",
                "booking_count",
                "booking_value",
                "successful_booking_value",
                "platform_revenue",
                "cancellations",
                "refunds",
                "experiment_variant",
                "alternatives_shown",
            ],
            "prediction_point": ("after first search results, before booking outcome"),
        }
        (MODEL_DIR / "stage6_model_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=json_default),
            encoding="utf-8",
        )

        write_documents(
            champion_name,
            champion_test,
            comparison,
            probit_test_metrics,
            cluster_metadata,
            permutation,
        )

        validation_report = {
            "status": "PASS",
            "data_classification": "synthetic portfolio data",
            "checks": checks,
            "row_counts": {
                "model_frame": int(len(frame)),
                "train": int(len(train)),
                "validation": int(len(validation)),
                "test": int(len(test)),
                "active_partners_clustered": int(len(clusters)),
            },
            "champion": champion_name,
            "locked_threshold": float(locked_threshold),
        }
        (OUT_DIR / "validation_report.json").write_text(
            json.dumps(validation_report, indent=2, default=json_default),
            encoding="utf-8",
        )

        print()
        print("=" * 78)
        print("RAILNEXUS - STAGE 6 PREDICTIVE MODELLING")
        print("=" * 78)
        print(f"Train rows                    : {len(train):,}")
        print(f"Validation rows               : {len(validation):,}")
        print(f"Locked test rows              : {len(test):,}")
        print(f"Champion                      : {champion_name}")
        print(f"Validation threshold          : {locked_threshold:.4f}")
        print(f"Test ROC-AUC                  : {champion_test['roc_auc']:.4f}")
        print(f"Test PR-AUC                   : {champion_test['pr_auc']:.4f}")
        print(f"Test top-decile lift          : {champion_test['top_decile_lift']:.3f}x")
        print(f"Probit test ROC-AUC           : {probit_test_metrics['roc_auc']:.4f}")
        print(f"Selected K-Means clusters     : {cluster_metadata['selected_k']}")
        print(f"Model card                    : {MODEL_CARD.relative_to(ROOT)}")
        print(f"Executive findings            : {FINDINGS_MD.relative_to(ROOT)}")
        print("STAGE 6                       : PASS")
        print("=" * 78)

    finally:
        con.close()


if __name__ == "__main__":
    main()
