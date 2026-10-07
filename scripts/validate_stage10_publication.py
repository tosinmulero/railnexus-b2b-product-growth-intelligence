from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "generated" / "stage10"

REQUIRED = [
    ROOT / "README.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "RECRUITER_SUMMARY.md",
    ROOT / "docs" / "CASE_STUDY.md",
    ROOT / "docs" / "EXPERIMENT_CARD.md",
    ROOT / "docs" / "STAGE6_MODEL_CARD.md",
    ROOT / "docs" / "AI_ASSISTED_INVESTIGATION.md",
    ROOT / "docs" / "API.md",
    ROOT / "docs" / "PRODUCTION_HARDENING.md",
    ROOT / ".github" / "workflows" / "ci.yml",
    ROOT / "Dockerfile",
    ROOT / "docker-compose.yml",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        raise RuntimeError("Publication files missing: " + ", ".join(missing))

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "No proprietary Trainline" not in readme:
        raise RuntimeError("README synthetic-data disclosure is missing.")

    report = {
        "status": "PASS",
        "publication_ready": True,
        "required_files": len(REQUIRED),
        "synthetic_data_disclosure": True,
        "architecture_documented": True,
        "experiment_documented": True,
        "model_documented": True,
        "responsible_ai_documented": True,
        "api_documented": True,
        "docker_documented": True,
        "ci_documented": True,
    }

    (OUT_DIR / "validation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("STAGE 10 PUBLICATION VALIDATION: PASS")


if __name__ == "__main__":
    main()
