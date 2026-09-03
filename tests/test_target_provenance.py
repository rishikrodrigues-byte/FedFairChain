from pathlib import Path
import csv
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def test_methodology_audit_runs_without_test_dataset_access():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "20_methodology_audit.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "METHODOLOGY AUDIT RESULT: PASS" in result.stdout
    assert "does not load data/processed/test.csv" in result.stdout


def test_target_provenance_report_has_no_direct_leakage():
    report = ROOT / "results" / "target_provenance_audit.csv"
    assert report.exists()
    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["target"] for row in rows} == {"DIABETES", "CVD", "HYPERTENSION"}
    assert all(row["direct_leakage_detected"].lower() == "false" for row in rows)


def test_chain_context_audit_confirms_teacher_forcing():
    report = ROOT / "results" / "chain_training_context_audit.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["test_dataset_loaded"] is False
    assert data["same_row_prediction_context_leakage_detected"] is False
    assert data["oof_correction_required_by_a2"] is False
    for script in data["scripts"].values():
        assert script["training_context"] == "ground_truth_previous_labels"
        assert script["classification"] == "A"
