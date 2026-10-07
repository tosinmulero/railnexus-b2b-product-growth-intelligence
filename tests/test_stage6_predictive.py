import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "railnexus.duckdb"
OUT_DIR = ROOT / "reports" / "generated" / "stage6"
MODEL_DIR = ROOT / "artifacts" / "models"


def connect() -> duckdb.DuckDBPyConnection:
    assert DB_PATH.exists()
    return duckdb.connect(str(DB_PATH), read_only=True)


def test_stage6_tables_exist() -> None:
    expected = {
        "mart_stage6_model_validation",
        "mart_stage6_test_scores",
        "mart_stage6_feature_importance",
        "mart_stage6_partner_clusters",
        "mart_stage6_cluster_profile",
    }

    with connect() as con:
        actual = {
            row[0]
            for row in con.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }

    assert expected.issubset(actual)


def test_validation_models_present() -> None:
    with connect() as con:
        models = {
            row[0]
            for row in con.execute(
                """
                SELECT model
                FROM mart_stage6_model_validation
                """
            ).fetchall()
        }

    assert {
        "Dummy",
        "LogisticRegression",
        "RandomForest",
        "MLP",
        "Probit",
    }.issubset(models)


def test_locked_threshold_comes_from_validation() -> None:
    metadata_path = MODEL_DIR / "stage6_model_metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["threshold_source"] == "validation"
    assert 0 <= metadata["locked_threshold"] <= 1
    assert metadata["test_start"].startswith("2026-05-01")


def test_champion_metrics_are_valid() -> None:
    path = OUT_DIR / "champion_test_metrics.json"
    assert path.exists()

    metrics = json.loads(path.read_text(encoding="utf-8"))

    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["brier"] <= 1
    assert metrics["top_decile_lift"] >= 0
    assert metrics["threshold_source"] == "validation"


def test_clusters_are_complete() -> None:
    with connect() as con:
        total, assigned, unique_clusters = con.execute(
            """
            SELECT
                COUNT(*),
                COUNT(cluster_id),
                COUNT(DISTINCT cluster_id)
            FROM mart_stage6_partner_clusters
            """
        ).fetchone()

    assert total > 0
    assert total == assigned
    assert unique_clusters >= 2


def test_stage6_artifacts_exist() -> None:
    expected = [
        MODEL_DIR / "stage6_booking_champion.joblib",
        MODEL_DIR / "stage6_partner_kmeans.joblib",
        MODEL_DIR / "stage6_model_metadata.json",
        ROOT / "docs" / "STAGE6_MODEL_CARD.md",
        ROOT / "reports" / "STAGE6_MODEL_FINDINGS.md",
        OUT_DIR / "validation_report.json",
        OUT_DIR / "probit_coefficients.csv",
        OUT_DIR / "champion_permutation_importance.csv",
    ]

    for path in expected:
        assert path.exists(), str(path)
