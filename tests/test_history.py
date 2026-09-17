from pathlib import Path

import pytest

from app.history import (
    delete_patient_history,
    get_patient_history,
    save_prediction,
    validate_patient_id,
)


def _prediction():
    return {
        "chain_order": ["DIABETES", "CVD", "HYPERTENSION"],
        "results": {
            "DIABETES": {
                "source": "predicted",
                "probability_percent": 76.58,
                "threshold_percent": 68.0,
                "primary_interpretation": "Above model threshold",
                "prediction": "Positive",
                "known_status": None,
            },
            "CVD": {
                "source": "predicted",
                "probability_percent": 53.57,
                "threshold_percent": 62.0,
                "primary_interpretation": "Below model threshold",
                "prediction": "Negative",
                "known_status": None,
            },
            "HYPERTENSION": {
                "source": "known",
                "probability_percent": None,
                "threshold_percent": None,
                "primary_interpretation": None,
                "prediction": None,
                "known_status": "Positive",
            },
        },
    }


def test_history_round_trip_and_delete(tmp_path: Path):
    db = tmp_path / "history.db"
    patient = {"AGE": 55, "BMI": 31.2, "HBA1C": 7.1}
    known = {"DIABETES": "unknown", "CVD": "unknown", "HYPERTENSION": "yes"}

    record_id = save_prediction("P001", patient, known, _prediction(), db_path=db)
    assert record_id == 1

    records = get_patient_history("P001", db_path=db)
    assert len(records) == 1
    assert records[0]["patient_id"] == "P001"
    assert records[0]["inputs"] == patient
    assert records[0]["known_diagnoses"] == known
    assert records[0]["results"]["DIABETES"]["probability_percent"] == 76.58
    assert records[0]["results"]["HYPERTENSION"]["known_status"] == "Positive"

    assert delete_patient_history("P001", db_path=db) == 1
    assert get_patient_history("P001", db_path=db) == []


def test_history_is_separated_by_anonymous_id(tmp_path: Path):
    db = tmp_path / "history.db"
    save_prediction("P001", {"AGE": 50}, {}, _prediction(), db_path=db)
    save_prediction("P002", {"AGE": 60}, {}, _prediction(), db_path=db)
    assert len(get_patient_history("P001", db_path=db)) == 1
    assert len(get_patient_history("P002", db_path=db)) == 1


@pytest.mark.parametrize("bad_id", ["", "John Doe", "123/45", "a" * 33, "abc@123"])
def test_invalid_patient_ids_are_rejected(bad_id):
    with pytest.raises(ValueError):
        validate_patient_id(bad_id)


def test_history_stores_compact_results_not_shap(tmp_path: Path):
    db = tmp_path / "history.db"
    prediction = _prediction()
    prediction["results"]["DIABETES"]["explanation"] = [{"shap_value": 2.18}]
    save_prediction("P001", {"AGE": 55}, {}, prediction, db_path=db)
    stored = get_patient_history("P001", db_path=db)[0]
    assert "explanation" not in stored["results"]["DIABETES"]


def test_ui_explains_history_is_optional_local_and_not_retraining():
    html = (Path(__file__).resolve().parents[1] / "app" / "index.html").read_text(encoding="utf-8")
    assert "Optional Prediction History" in html
    assert "Anonymous Patient ID" in html
    assert "Save this prediction locally" in html
    assert "not used to retrain FedFairChain" in html
    assert "View Saved History" in html
    assert "Load these inputs" in html
