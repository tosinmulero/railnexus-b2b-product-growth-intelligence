import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_recruiter_sections() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    required = [
        "B2B Rail Product Growth",
        "Product experiment",
        "Predictive modelling",
        "Network opportunity",
        "Architecture",
        "Run the API",
        "Run with Docker",
        "Data notice",
    ]

    for item in required:
        assert item in text


def test_readme_discloses_synthetic_data() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "synthetic" in text
    assert "no proprietary trainline" in text


def test_architecture_contains_mermaid() -> None:
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "```mermaid" in text
    assert "FastAPI" in text
    assert "Human Review" in text


def test_recruiter_and_case_study_docs_exist() -> None:
    for path in [
        ROOT / "docs" / "RECRUITER_SUMMARY.md",
        ROOT / "docs" / "CASE_STUDY.md",
    ]:
        assert path.exists()
        assert len(path.read_text(encoding="utf-8")) > 500


def test_publication_manifest() -> None:
    path = ROOT / "reports" / "generated" / "stage10" / "publication_manifest.json"
    assert path.exists()

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["data_classification"] == "synthetic portfolio data"
    assert report["validation_threshold_source"] == "validation"
