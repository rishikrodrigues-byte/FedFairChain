import pandas as pd
import pytest

from src.contracts.feature_contract import (
    BASE_FEATURES,
    CHAIN_ORDER,
    TARGET_COLUMNS,
    get_base_features,
    get_chain_order,
    get_stage_features,
    validate_base_dataframe,
    validate_chain_order,
    validate_stage_features,
)


def test_exact_base_feature_count_and_uniqueness():
    assert len(BASE_FEATURES) == 15
    assert len(BASE_FEATURES) == len(set(BASE_FEATURES))


def test_protected_attribute_and_targets_excluded():
    assert "SEX" not in BASE_FEATURES
    assert not TARGET_COLUMNS.intersection(BASE_FEATURES)


def test_chain_order_locked():
    assert CHAIN_ORDER == ("DIABETES", "CVD", "HYPERTENSION")
    assert get_chain_order() == ["DIABETES", "CVD", "HYPERTENSION"]
    validate_chain_order(CHAIN_ORDER)
    with pytest.raises(ValueError):
        validate_chain_order(["CVD", "DIABETES", "HYPERTENSION"])


def test_stage_dimensions_and_names():
    diabetes = get_stage_features("DIABETES")
    cvd = get_stage_features("CVD")
    hypertension = get_stage_features("HYPERTENSION")
    assert len(diabetes) == 15
    assert len(cvd) == 16
    assert len(hypertension) == 17
    assert cvd[-1] == "DIABETES_CONTEXT"
    assert hypertension[-2:] == ["DIABETES_CONTEXT", "CVD_CONTEXT"]


def test_invalid_stage_rejected():
    with pytest.raises(ValueError):
        get_stage_features("KIDNEY")


def test_deterministic_feature_order():
    assert get_base_features() == list(BASE_FEATURES)
    assert get_base_features() == get_base_features()


def test_validate_base_dataframe_accepts_exact_contract():
    df = pd.DataFrame([[0] * 15], columns=list(BASE_FEATURES))
    validate_base_dataframe(df)


def test_validate_base_dataframe_rejects_wrong_order():
    cols = list(BASE_FEATURES)
    cols[0], cols[1] = cols[1], cols[0]
    df = pd.DataFrame([[0] * 15], columns=cols)
    with pytest.raises(ValueError):
        validate_base_dataframe(df)


def test_validate_stage_features_rejects_mismatch():
    validate_stage_features("CVD", get_stage_features("CVD"))
    with pytest.raises(ValueError):
        validate_stage_features("CVD", get_stage_features("DIABETES"))
