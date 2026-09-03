from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from app import predictor
from src.contracts.feature_contract import (
    get_base_features,
    get_chain_order,
    get_stage_features,
)


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

# Exact pre-refactor baseline captured from the existing production predictor.
BASELINE_PROBABILITIES = {
    "DIABETES": 0.7657755712505416,
    "CVD": 0.5356604109997184,
    "HYPERTENSION": 0.8777268246104308,
}


def test_loaded_artifact_matches_locked_contract():
    assert predictor.FEATURES == get_base_features()
    assert predictor.ORDER == get_chain_order()
    assert "SEX" not in predictor.FEATURES
    assert len(predictor.FEATURES) == 15

    assert len(get_stage_features("DIABETES")) == 15
    assert len(get_stage_features("CVD")) == 16
    assert len(get_stage_features("HYPERTENSION")) == 17

    for target in predictor.ORDER:
        assert predictor.MODELS[target].n_features_in_ == len(
            get_stage_features(target)
        )


def test_prediction_regression_and_input_immutability():
    patient = deepcopy(SAMPLE_PATIENT)
    before = deepcopy(patient)

    output = predictor.predict_patient(patient)

    assert patient == before
    assert output["chain_order"] == get_chain_order()
    assert output["features_used"] == get_base_features()
    assert output["sex_used_as_predictor"] is False
    assert set(output["results"]) == set(get_chain_order())

    for target, expected_probability in BASELINE_PROBABILITIES.items():
        result = output["results"][target]
        probability = result["probability"]

        assert result["source"] == "predicted"
        assert np.isfinite(probability)
        assert 0.0 <= probability <= 1.0
        assert probability == pytest.approx(expected_probability, abs=1e-12)


def test_known_diabetes_is_not_repredicted_and_is_propagated():
    output = predictor.predict_patient(
        deepcopy(SAMPLE_PATIENT),
        known_diagnoses={
            "DIABETES": "yes",
            "CVD": "unknown",
            "HYPERTENSION": "unknown",
        },
    )

    diabetes = output["results"]["DIABETES"]
    cvd = output["results"]["CVD"]

    assert diabetes["source"] == "known"
    assert diabetes["known_status"] == "Positive"
    assert diabetes["probability"] is None
    assert diabetes["prediction"] is None
    assert diabetes["explanation"] == []

    assert cvd["context_used"][0] == {
        "target": "DIABETES",
        "value": 1,
        "source": "known",
    }
