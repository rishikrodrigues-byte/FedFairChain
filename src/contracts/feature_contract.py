from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# Authoritative feature order verified against the existing trained
# fair_federated_classifier_chain.joblib artifact before this contract was added.
BASE_FEATURES = (
    "AGE",
    "BMI",
    "WAIST_CM",
    "SBP_MEAN",
    "DBP_MEAN",
    "HBA1C",
    "TOTAL_CHOLESTEROL",
    "HDL_CHOLESTEROL",
    "CREATININE",
    "BUN",
    "ALBUMIN",
    "URIC_ACID",
    "SERUM_GLUCOSE",
    "SMOKING_STATUS",
    "PHYSICAL_ACTIVITY_ANY",
)

CHAIN_ORDER = (
    "DIABETES",
    "CVD",
    "HYPERTENSION",
)

TARGET_COLUMNS = frozenset({
    "DIABETES",
    "CVD",
    "HYPERTENSION",
})

PROTECTED_ATTRIBUTE = "SEX"


def get_base_features() -> list[str]:
    return list(BASE_FEATURES)


def get_chain_order() -> list[str]:
    return list(CHAIN_ORDER)


def _validate_stage(stage: str) -> int:
    if stage not in CHAIN_ORDER:
        raise ValueError(
            f"Invalid classifier-chain stage {stage!r}. "
            f"Expected one of {list(CHAIN_ORDER)}."
        )
    return CHAIN_ORDER.index(stage)


def get_stage_features(stage: str) -> list[str]:
    position = _validate_stage(stage)
    names = list(BASE_FEATURES)
    for previous_target in CHAIN_ORDER[:position]:
        names.append(f"{previous_target}_CONTEXT")
    return names


def validate_chain_order(order: Iterable[str]) -> None:
    observed = tuple(order)
    if observed != CHAIN_ORDER:
        raise ValueError(
            "Classifier-chain order mismatch. "
            f"Expected {list(CHAIN_ORDER)}, got {list(observed)}."
        )


def validate_stage_features(stage: str, feature_names: Iterable[str]) -> None:
    expected = get_stage_features(stage)
    observed = list(feature_names)
    if observed != expected:
        raise ValueError(
            f"Feature contract mismatch for {stage}. "
            f"Expected {expected}, got {observed}."
        )


def validate_base_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    columns = list(df.columns)

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate dataframe columns are not allowed.")

    if PROTECTED_ATTRIBUTE in columns:
        raise ValueError("SEX must not be included in the disease-prediction feature dataframe.")

    target_overlap = TARGET_COLUMNS.intersection(columns)
    if target_overlap:
        raise ValueError(
            f"Target columns must not be included as base predictors: {sorted(target_overlap)}"
        )

    if columns != list(BASE_FEATURES):
        missing = [name for name in BASE_FEATURES if name not in columns]
        extra = [name for name in columns if name not in BASE_FEATURES]
        raise ValueError(
            "Base feature dataframe does not match the locked feature contract. "
            f"Missing={missing}; extra={extra}; observed_order={columns}"
        )
