from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ``python app\predictor.py`` places ``app`` on sys.path rather than the
# repository root. Add the root explicitly so the locked ML contract under
# src/contracts is available both for direct smoke tests and FastAPI imports.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.contracts.feature_contract import (
    get_base_features,
    get_chain_order,
    get_stage_features,
    validate_base_dataframe,
    validate_chain_order,
    validate_stage_features,
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "fair_federated_classifier_chain.joblib"
)

SHAP_PATH = (
    BASE_DIR
    / "models"
    / "shap_backgrounds.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_BUNDLE = joblib.load(
    MODEL_PATH
)

SHAP_BUNDLE = joblib.load(
    SHAP_PATH
)

FEATURES = list(
    MODEL_BUNDLE["features"]
)

ORDER = list(
    MODEL_BUNDLE["order"]
)

MODELS = MODEL_BUNDLE["models"]

THRESHOLDS = MODEL_BUNDLE[
    "thresholds"
]

PREPROCESSOR = MODEL_BUNDLE[
    "preprocessor"
]

SHAP_BACKGROUNDS = SHAP_BUNDLE[
    "backgrounds"
]

SHAP_FEATURE_NAMES = SHAP_BUNDLE[
    "feature_names"
]


# ============================================================
# LOCKED ARTIFACT / FEATURE-CONTRACT CHECK
# ============================================================

def _validate_loaded_artifacts():
    """Fail fast if saved artifacts do not match the locked ML contract."""

    expected_features = get_base_features()
    expected_order = get_chain_order()

    validate_chain_order(ORDER)

    if FEATURES != expected_features:
        raise RuntimeError(
            "Saved model base-feature contract mismatch. "
            f"Expected {expected_features}, got {FEATURES}."
        )

    if ORDER != expected_order:
        raise RuntimeError(
            "Saved model chain-order contract mismatch. "
            f"Expected {expected_order}, got {ORDER}."
        )

    if "SEX" in FEATURES:
        raise RuntimeError(
            "SEX must not be used as a prediction feature."
        )

    preprocessor_names = getattr(
        PREPROCESSOR,
        "feature_names_in_",
        None,
    )
    if preprocessor_names is not None:
        observed = list(preprocessor_names)
        if observed != expected_features:
            raise RuntimeError(
                "Saved preprocessor feature order does not match the "
                f"locked contract. Got {observed}."
            )

    for target in ORDER:
        if target not in MODELS:
            raise RuntimeError(
                f"Saved model bundle is missing classifier {target}."
            )

        stage_features = get_stage_features(target)
        model_n_features = getattr(
            MODELS[target],
            "n_features_in_",
            None,
        )
        if model_n_features is not None and int(model_n_features) != len(stage_features):
            raise RuntimeError(
                f"{target} classifier expects {model_n_features} features; "
                f"locked contract requires {len(stage_features)}."
            )

        if target not in SHAP_FEATURE_NAMES:
            raise RuntimeError(
                f"SHAP metadata is missing feature names for {target}."
            )

        validate_stage_features(
            target,
            SHAP_FEATURE_NAMES[target],
        )


_validate_loaded_artifacts()


# ============================================================
# KNOWN DIAGNOSIS CONVERSION
# ============================================================

def normalize_known_status(value):
    """
    Converts diagnosis input into:

    1    = known positive
    0    = known negative
    None = unknown
    """

    if value is None:
        return None

    if isinstance(value, str):

        value = (
            value.strip()
            .lower()
        )

        if value in {
            "yes",
            "positive",
            "1",
        }:
            return 1

        if value in {
            "no",
            "negative",
            "0",
        }:
            return 0

        if value in {
            "unknown",
            "",
        }:
            return None

    if value == 1:
        return 1

    if value == 0:
        return 0

    raise ValueError(
        "Diagnosis status must be "
        "Yes, No or Unknown."
    )


# ============================================================
# PREPARE PATIENT FEATURES
# ============================================================

def prepare_patient(patient):
    """
    Builds one-row dataframe in the exact feature
    order used during training.
    """

    row = {}

    for feature in FEATURES:

        value = patient.get(
            feature
        )

        if value is None:
            value = np.nan

        row[feature] = value

    patient_df = pd.DataFrame(
        [row],
        columns=FEATURES,
    )

    # This validates feature names and, critically, their order. Missing
    # numeric values are still permitted here and are handled by the frozen
    # training-fitted preprocessing pipeline.
    validate_base_dataframe(patient_df)

    return patient_df


# ============================================================
# SHAP LOCAL EXPLANATION
# ============================================================

def explain_prediction(
    target,
    X_current,
    top_n=5,
):

    model = MODELS[target]

    background = (
        SHAP_BACKGROUNDS[target]
    )

    feature_names = list(
        SHAP_FEATURE_NAMES[target]
    )

    explainer = shap.LinearExplainer(
        model,
        background,
    )

    explanation = explainer(
        X_current
    )

    values = np.asarray(
        explanation.values
    )[0]

    rows = []

    for feature, value in zip(
        feature_names,
        values,
    ):

        rows.append(
            {
                "feature": feature,
                "shap_value": float(
                    value
                ),
                "absolute_shap": float(
                    abs(value)
                ),
                "direction": (
                    "increases"
                    if value > 0
                    else "decreases"
                ),
            }
        )

    rows.sort(
        key=lambda item:
        item["absolute_shap"],
        reverse=True,
    )

    return rows[:top_n]


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_patient(
    patient,
    known_diagnoses=None,
    top_n_explanations=5,
):
    """
    Diagnosis-aware FedFairChain inference.

    patient:
        dictionary containing the 15 features.

    known_diagnoses:
        {
            "DIABETES": "yes/no/unknown",
            "CVD": "yes/no/unknown",
            "HYPERTENSION": "yes/no/unknown"
        }
    """

    if known_diagnoses is None:
        known_diagnoses = {}

    known = {
        target:
        normalize_known_status(
            known_diagnoses.get(
                target,
                "unknown",
            )
        )
        for target in ORDER
    }


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    patient_df = prepare_patient(
        patient
    )

    X_current = (
        PREPROCESSOR.transform(
            patient_df
        )
    )


    results = {}

    chain_context = []


    # ========================================================
    # SEQUENTIAL CLASSIFIER CHAIN
    # ========================================================

    for target in ORDER:

        model = MODELS[target]

        threshold = float(
            THRESHOLDS[target]
        )

        known_value = known[
            target
        ]


        # ----------------------------------------------------
        # KNOWN DIAGNOSIS
        # ----------------------------------------------------

        if known_value is not None:

            chain_value = int(
                known_value
            )

            results[target] = {
                "source":
                    "known",

                "known_status":
                    (
                        "Positive"
                        if chain_value == 1
                        else "Negative"
                    ),

                "probability":
                    None,

                "prediction":
                    None,

                "threshold":
                    threshold,

                "explanation":
                    [],

                "message":
                    (
                        "Known diagnosis supplied. "
                        "No model prediction was "
                        "reported for this condition."
                    ),

                "context_used":
                    list(
                        chain_context
                    ),
            }


        # ----------------------------------------------------
        # UNKNOWN → MODEL PREDICTS
        # ----------------------------------------------------

        else:

            probability = float(
                model.predict_proba(
                    X_current
                )[0, 1]
            )

            prediction = int(
                probability
                >= threshold
            )

            chain_value = (
                prediction
            )

            explanation = (
                explain_prediction(
                    target,
                    X_current,
                    top_n=(
                        top_n_explanations
                    ),
                )
            )

            results[target] = {
                "source":
                    "predicted",

                "known_status":
                    None,

                "probability":
                    probability,

                "probability_percent":
                    round(
                        probability * 100,
                        2,
                    ),

                "prediction":
                    (
                        "Positive"
                        if prediction == 1
                        else "Negative"
                    ),

                "prediction_value":
                    prediction,

                "threshold":
                    threshold,

                "explanation":
                    explanation,

                "context_used":
                    list(
                        chain_context
                    ),
            }


        # ----------------------------------------------------
        # ADD VALUE TO CLASSIFIER CHAIN
        # ----------------------------------------------------

        X_current = np.column_stack(
            [
                X_current,
                chain_value,
            ]
        )


        chain_context.append(
            {
                "target":
                    target,

                "value":
                    chain_value,

                "source":
                    (
                        "known"
                        if known_value
                        is not None
                        else "predicted"
                    ),
            }
        )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {
        "model":
            "FedFairChain",

        "chain_order":
            ORDER,

        "features_used":
            FEATURES,

        "sex_used_as_predictor":
            False,

        "results":
            results,

        "final_chain_context":
            chain_context,

        "disclaimer":
            (
                "Research prototype only. "
                "Predictions are not a medical "
                "diagnosis and should not replace "
                "professional clinical evaluation."
            ),
    }


# ============================================================
# LOCAL SMOKE TEST
# ============================================================

if __name__ == "__main__":

    sample_patient = {

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


    sample_known = {
    "DIABETES": "unknown",
    "CVD": "unknown",
    "HYPERTENSION": "unknown",
}


    output = predict_patient(
        sample_patient,
        sample_known,
    )


    print(
        "\n"
        + "=" * 75
    )

    print(
        "FEDFAIRCHAIN PREDICTION TEST"
    )

    print(
        "=" * 75
    )


    print(
        "\nChain:"
    )

    print(
        " -> ".join(
            output[
                "chain_order"
            ]
        )
    )


    for target in ORDER:

        result = (
            output[
                "results"
            ][target]
        )

        print(
            "\n"
            + "-" * 75
        )

        print(target)

        print(
            "-" * 75
        )


        if (
            result["source"]
            == "known"
        ):

            print(
                "Source      : "
                "KNOWN DIAGNOSIS"
            )

            print(
                "Status      : "
                + result[
                    "known_status"
                ]
            )


        else:

            print(
                "Source      : "
                "MODEL PREDICTION"
            )

            print(
                f"Probability : "
                f"{result['probability_percent']:.2f}%"
            )

            print(
                "Prediction  : "
                + result[
                    "prediction"
                ]
            )

            print(
                f"Threshold   : "
                f"{result['threshold']:.2f}"
            )


            print(
                "\nTop contributors:"
            )

            for item in (
                result[
                    "explanation"
                ]
            ):

                print(
                    f"  "
                    f"{item['feature']:<25} "
                    f"{item['shap_value']:+.4f} "
                    f"({item['direction']})"
                )


        print(
            "\nPrevious chain context:"
        )

        if not result[
            "context_used"
        ]:

            print(
                "  None"
            )

        else:

            for context in (
                result[
                    "context_used"
                ]
            ):

                print(
                    f"  "
                    f"{context['target']}="
                    f"{context['value']} "
                    f"({context['source']})"
                )


    print(
        "\n"
        + "=" * 75
    )

    print(
        output["disclaimer"]
    )

    print(
        "=" * 75
    )