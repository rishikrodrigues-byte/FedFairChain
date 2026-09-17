from pathlib import Path
import hashlib
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

FEATURE_DISPLAY_NAMES = {
    "AGE": "Age",
    "BMI": "BMI",
    "WAIST_CM": "Waist Circumference",
    "SBP_MEAN": "Systolic Blood Pressure",
    "DBP_MEAN": "Diastolic Blood Pressure",
    "HBA1C": "HbA1c",
    "TOTAL_CHOLESTEROL": "Total Cholesterol",
    "HDL_CHOLESTEROL": "HDL Cholesterol",
    "CREATININE": "Creatinine",
    "BUN": "BUN",
    "ALBUMIN": "Albumin",
    "URIC_ACID": "Uric Acid",
    "SERUM_GLUCOSE": "Serum Glucose",
    "SMOKING_STATUS": "Smoking Status",
    "PHYSICAL_ACTIVITY_ANY": "Physical Activity",
    "DIABETES_CONTEXT": "Diabetes chain context",
    "CVD_CONTEXT": "CVD chain context",
}

FEATURE_UNITS = {
    "AGE": "years",
    "BMI": "kg/m²",
    "WAIST_CM": "cm",
    "SBP_MEAN": "mmHg",
    "DBP_MEAN": "mmHg",
    "HBA1C": "%",
    "TOTAL_CHOLESTEROL": "mg/dL",
    "HDL_CHOLESTEROL": "mg/dL",
    "CREATININE": "mg/dL",
    "BUN": "mg/dL",
    "ALBUMIN": "g/dL",
    "URIC_ACID": "mg/dL",
    "SERUM_GLUCOSE": "mg/dL",
}


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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MODEL_SHA256 = _sha256_file(MODEL_PATH)

# SHAP explainers are relatively expensive to construct. Build them once when
# the application starts and reuse them for every inference request.
SHAP_EXPLAINERS = {
    target: shap.LinearExplainer(
        MODELS[target],
        SHAP_BACKGROUNDS[target],
    )
    for target in ORDER
}


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

def _relative_strength(rank):
    if rank == 1:
        return "Very strong influence"
    if rank == 2:
        return "Strong influence"
    if rank == 3:
        return "Moderate influence"
    return "Smaller influence"


def _display_value(feature, patient, context_used):
    if feature.endswith("_CONTEXT"):
        target = feature.removesuffix("_CONTEXT")
        item = next(
            (entry for entry in context_used if entry["target"] == target),
            None,
        )
        if item is None:
            return None, "Not available", None
        value = int(item["value"])
        return value, ("Positive" if value == 1 else "Negative"), None

    value = patient.get(feature)
    if value is None:
        return None, "Unavailable — training-fitted imputation used", FEATURE_UNITS.get(feature)

    if feature == "SMOKING_STATUS":
        labels = {0: "Never smoker", 1: "Former smoker", 2: "Current smoker"}
        return value, labels.get(value, str(value)), None

    if feature == "PHYSICAL_ACTIVITY_ANY":
        labels = {0: "No", 1: "Yes"}
        return value, labels.get(value, str(value)), None

    unit = FEATURE_UNITS.get(feature)
    return value, f"{value:g}" if isinstance(value, (int, float)) else str(value), unit


def explain_all_features(target, X_current, patient, context_used):
    feature_names = list(SHAP_FEATURE_NAMES[target])
    explanation = SHAP_EXPLAINERS[target](X_current)
    values = np.asarray(explanation.values)[0]

    rows = []
    for feature, value in zip(feature_names, values):
        raw_value, display_value, unit = _display_value(
            feature, patient, context_used
        )
        rows.append({
            "feature": feature,
            "display_name": FEATURE_DISPLAY_NAMES.get(feature, feature),
            "shap_value": float(value),
            "absolute_shap": float(abs(value)),
            "direction": "increases" if value > 0 else "decreases",
            "direction_label": "upward" if value > 0 else "downward",
            "is_chain_context": feature.endswith("_CONTEXT"),
            "patient_value": raw_value,
            "patient_value_display": display_value,
            "unit": unit,
        })

    rows.sort(key=lambda item: item["absolute_shap"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["relative_rank"] = rank
        row["relative_strength"] = _relative_strength(rank)

    return rows


def explain_prediction(
    target,
    X_current,
    top_n=5,
    patient=None,
    context_used=None,
):
    patient = {} if patient is None else patient
    context_used = [] if context_used is None else context_used
    return explain_all_features(
        target, X_current, patient, context_used
    )[:top_n]


def get_model_info():
    return {
        "project": "FedFairChain",
        "model_name": "Fairness-aware federated classifier chain",
        "model_hash_sha256": MODEL_SHA256,
        "chain_order": list(ORDER),
        "context_type": "hard_label",
        "base_feature_count": len(FEATURES),
        "thresholds": {key: float(value) for key, value in THRESHOLDS.items()},
        "federated_clients": int(MODEL_BUNDLE.get("num_clients", 4)),
        "federated_environment": "simulated",
        "federated_rounds": int(MODEL_BUNDLE.get("federated_rounds", 20)),
        "best_round": int(MODEL_BUNDLE.get("best_round", 16)),
        "local_epochs": int(MODEL_BUNDLE.get("local_epochs", 2)),
        "aggregation": "FedAvg",
        "fairness_aware": True,
        "fairness_method": MODEL_BUNDLE.get("fairness_method", "Sex-label reweighing"),
        "protected_attribute": MODEL_BUNDLE.get("protected_attribute", "SEX"),
        "protected_attribute_used_for_prediction": False,
        "diagnosis_aware": True,
        "explainability": "SHAP",
        "intended_use": "Research and educational prototype",
    }


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

    provided_features = [
        feature for feature in FEATURES
        if patient.get(feature) is not None
    ]
    missing_features = [
        feature for feature in FEATURES
        if patient.get(feature) is None
    ]
    input_completeness = {
        "provided_count": len(provided_features),
        "total_count": len(FEATURES),
        "missing_count": len(missing_features),
        "provided_features": provided_features,
        "missing_features": missing_features,
        "message": (
            "All model features were supplied."
            if not missing_features
            else (
                f"{len(provided_features)} of {len(FEATURES)} model features were supplied. "
                "Unavailable values were handled by the frozen training-fitted imputation pipeline."
            )
        ),
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

                "disease_stage": {
                    "estimated": False,
                    "message": (
                        "FedFairChain does not estimate clinical disease stage or severity."
                    ),
                },

                "next_steps": [
                    "Continue following the care and monitoring plan provided by your healthcare professional.",
                    "Do not start, stop, or change medication based on this application.",
                ],
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

            all_explanation = explain_all_features(
                target,
                X_current,
                patient,
                list(chain_context),
            )
            explanation = all_explanation[:top_n_explanations]
            clinical_contributors = [
                item for item in all_explanation
                if not item["is_chain_context"]
            ][:top_n_explanations]
            context_contributors = [
                item for item in all_explanation
                if item["is_chain_context"]
            ]

            enriched_context = []
            for context_item in chain_context:
                context_feature = f"{context_item['target']}_CONTEXT"
                influence = next(
                    (item for item in context_contributors if item["feature"] == context_feature),
                    None,
                )
                enriched = dict(context_item)
                if influence is not None:
                    enriched.update({
                        "influence_direction": influence["direction_label"],
                        "relative_strength": influence["relative_strength"],
                        "shap_value": influence["shap_value"],
                    })
                enriched_context.append(enriched)

            threshold_difference = probability - threshold
            threshold_position = "above" if prediction == 1 else "below"

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

                "threshold_percent":
                    round(threshold * 100, 2),

                "threshold_difference":
                    float(threshold_difference),

                "threshold_difference_percent":
                    round(threshold_difference * 100, 2),

                "threshold_position":
                    threshold_position,

                "primary_interpretation":
                    (
                        "Above model threshold"
                        if prediction == 1
                        else "Below model threshold"
                    ),

                "explanation":
                    explanation,

                "clinical_contributors":
                    clinical_contributors,

                "chain_context_contributors":
                    context_contributors,

                "context_used":
                    list(chain_context),

                "context_explanations":
                    enriched_context,

                "disease_stage": {
                    "estimated": False,
                    "message": (
                        "FedFairChain does not estimate clinical disease stage or severity."
                    ),
                },

                "next_steps": (
                    [
                        "Consider discussing this above-threshold model result with a qualified healthcare professional.",
                        "A healthcare professional can determine whether clinical assessment or confirmatory testing is appropriate.",
                        "Do not start, stop, or change medication based on this application.",
                    ]
                    if prediction == 1
                    else [
                        "A below-threshold model result does not rule out disease.",
                        "Continue routine health checks and seek professional assessment when appropriate.",
                        "Do not start, stop, or change medication based on this application.",
                    ]
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

    summary_items = []
    for target in ORDER:
        result = results[target]
        if result["source"] == "known":
            status = f"Known {result['known_status'].lower()} diagnosis supplied"
        else:
            status = result["primary_interpretation"]
        summary_items.append({
            "target": target,
            "status": status,
            "source": result["source"],
        })

    summary_text = "; ".join(
        f"{item['target'].title()}: {item['status']}"
        for item in summary_items
    )

    return {
        "model":
            "FedFairChain",

        "chain_order":
            ORDER,

        "features_used":
            FEATURES,

        "sex_used_as_predictor":
            False,

        "model_hash_sha256":
            MODEL_SHA256,

        "input_completeness":
            input_completeness,

        "model_summary": {
            "items": summary_items,
            "text": summary_text,
            "safe_percentage_note": (
                "There is no universal 'safe percentage' in this model. "
                "Each disease probability is compared with its own learned decision threshold, "
                "which is a model cutoff and not a medical safety limit."
            ),
        },

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