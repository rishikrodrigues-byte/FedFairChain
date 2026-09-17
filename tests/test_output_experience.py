from copy import deepcopy

from app import predictor


SAMPLE_PATIENT = {
    "AGE": 55,
    "BMI": 31.2,
    "WAIST_CM": 104,
    "SBP_MEAN": 148,
    "DBP_MEAN": 92,
    "HBA1C": 7.1,
    "TOTAL_CHOLESTEROL": 215,
    "HDL_CHOLESTEROL": 42,
    "CREATININE": 0.95,
    "BUN": 16,
    "ALBUMIN": 4.1,
    "URIC_ACID": 6.2,
    "SERUM_GLUCOSE": 142,
    "SMOKING_STATUS": 1,
    "PHYSICAL_ACTIVITY_ANY": 1,
}


def test_beginner_friendly_prediction_metadata():
    output = predictor.predict_patient(deepcopy(SAMPLE_PATIENT))
    assert output["input_completeness"]["provided_count"] == 15
    assert "safe percentage" in output["model_summary"]["safe_percentage_note"].lower()

    for target in output["chain_order"]:
        result = output["results"][target]
        assert result["source"] == "predicted"
        assert result["threshold_position"] in {"above", "below"}
        assert result["primary_interpretation"] in {
            "Above model threshold",
            "Below model threshold",
        }
        assert result["disease_stage"]["estimated"] is False
        assert "does not estimate" in result["disease_stage"]["message"].lower()
        assert len(result["next_steps"]) >= 2
        assert result["clinical_contributors"]

        for item in result["clinical_contributors"]:
            assert item["relative_strength"] in {
                "Very strong influence",
                "Strong influence",
                "Moderate influence",
                "Smaller influence",
            }
            assert item["direction_label"] in {"upward", "downward"}
            assert item["is_chain_context"] is False


def test_missing_input_completeness_is_truthful():
    patient = deepcopy(SAMPLE_PATIENT)
    patient["CREATININE"] = None
    patient["BUN"] = None
    patient["ALBUMIN"] = None
    output = predictor.predict_patient(patient)
    completeness = output["input_completeness"]
    assert completeness["provided_count"] == 12
    assert completeness["missing_count"] == 3
    assert completeness["missing_features"] == ["CREATININE", "BUN", "ALBUMIN"]


def test_known_diagnosis_has_no_probability_or_shap():
    output = predictor.predict_patient(
        deepcopy(SAMPLE_PATIENT),
        known_diagnoses={
            "DIABETES": "yes",
            "CVD": "unknown",
            "HYPERTENSION": "unknown",
        },
    )
    diabetes = output["results"]["DIABETES"]
    assert diabetes["source"] == "known"
    assert diabetes["probability"] is None
    assert diabetes["explanation"] == []
    assert diabetes["disease_stage"]["estimated"] is False


def test_model_info_exposes_safe_metadata_only():
    info = predictor.get_model_info()
    assert info["project"] == "FedFairChain"
    assert info["federated_environment"] == "simulated"
    assert info["federated_clients"] == 4
    assert info["protected_attribute_used_for_prediction"] is False
    assert info["model_hash_sha256"] == predictor.MODEL_SHA256
    assert len(info["model_hash_sha256"]) == 64
    forbidden = {"SEQN", "patient_rows", "train_data", "test_data", "client_dataset"}
    assert forbidden.isdisjoint(info)
