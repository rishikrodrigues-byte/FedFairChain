from pathlib import Path
from itertools import combinations

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

VAL_PATH = Path("data/processed/validation.csv")

MODEL_PATH = Path(
    "models/best_classifier_chain.joblib"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "DIABETES",
    "HYPERTENSION",
    "CVD",
]


# ============================================================
# LOAD VALIDATION DATA + BEST CHAIN
# ============================================================

print("=" * 82)
print("FEDFAIRCHAIN - DIAGNOSIS-AWARE CLASSIFIER CHAIN")
print("=" * 82)

val_df = pd.read_csv(VAL_PATH)

bundle = joblib.load(MODEL_PATH)

features = bundle["features"]
order = list(bundle["order"])
thresholds = bundle["thresholds"]
preprocessor = bundle["preprocessor"]
models = bundle["models"]


print(f"\nValidation rows : {len(val_df)}")
print(f"Predictors      : {len(features)}")
print(f"Chain order     : {' -> '.join(order)}")

print("\nKnown diagnosis simulation:")
print(
    "Ground-truth diagnosis values are supplied ONLY "
    "for labels declared as known."
)
print(
    "Known labels are excluded from evaluation."
)

print("\n[PASS] TEST dataset is not loaded.")


# ============================================================
# PREPROCESS VALIDATION FEATURES
# ============================================================

X_base = preprocessor.transform(
    val_df[features]
)


# ============================================================
# NORMAL CLASSIFIER-CHAIN INFERENCE
# ============================================================

def standard_chain_predict():

    probabilities = {}
    predictions = {}

    X_current = X_base.copy()

    for target in order:

        model = models[target]

        probability = model.predict_proba(
            X_current
        )[:, 1]

        threshold = thresholds[target]

        prediction = (
            probability >= threshold
        ).astype(int)

        probabilities[target] = probability
        predictions[target] = prediction

        X_current = np.column_stack(
            [
                X_current,
                prediction,
            ]
        )

    return probabilities, predictions


# ============================================================
# DIAGNOSIS-AWARE INFERENCE
# ============================================================

def diagnosis_aware_predict(
    known_targets,
):

    """
    known_targets:
        A set containing labels already known
        by the patient.

    During this validation experiment,
    the true diagnosis is used as the known
    patient-provided value.

    Known labels are NOT scored.
    """

    probabilities = {}
    predictions = {}

    X_current = X_base.copy()

    for target in order:

        # ----------------------------------------------------
        # KNOWN DIAGNOSIS
        # ----------------------------------------------------

        if target in known_targets:

            known_value = (
                val_df[target]
                .astype(int)
                .to_numpy()
            )

            # The confirmed diagnosis becomes
            # classifier-chain context.
            predictions[target] = known_value

            # Only stored for completeness.
            probabilities[target] = (
                known_value.astype(float)
            )

            context_value = known_value

        # ----------------------------------------------------
        # UNKNOWN DIAGNOSIS
        # ----------------------------------------------------

        else:

            model = models[target]

            probability = model.predict_proba(
                X_current
            )[:, 1]

            threshold = thresholds[target]

            prediction = (
                probability >= threshold
            ).astype(int)

            probabilities[target] = probability
            predictions[target] = prediction

            context_value = prediction

        # ----------------------------------------------------
        # PASS LABEL CONTEXT FORWARD
        # ----------------------------------------------------

        X_current = np.column_stack(
            [
                X_current,
                context_value,
            ]
        )

    return probabilities, predictions


# ============================================================
# METRICS
# ============================================================

def target_metrics(
    y_true,
    probability,
    prediction,
):

    return {
        "roc_auc":
            roc_auc_score(
                y_true,
                probability
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probability
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                prediction
            ),
    }


# ============================================================
# STANDARD CHAIN PREDICTIONS
# ============================================================

standard_probabilities, standard_predictions = (
    standard_chain_predict()
)


# ============================================================
# SCENARIO EVALUATION
# ============================================================

detail_rows = []
summary_rows = []


def evaluate_scenario(
    scenario_name,
    known_targets,
):

    known_targets = set(known_targets)

    unknown_targets = [
        target
        for target in TARGETS
        if target not in known_targets
    ]

    da_probabilities, da_predictions = (
        diagnosis_aware_predict(
            known_targets
        )
    )

    scenario_details = []

    for target in unknown_targets:

        y_true = (
            val_df[target]
            .astype(int)
            .to_numpy()
        )

        # Standard chain
        standard = target_metrics(
            y_true,
            standard_probabilities[target],
            standard_predictions[target],
        )

        # Diagnosis-aware chain
        diagnosis_aware = target_metrics(
            y_true,
            da_probabilities[target],
            da_predictions[target],
        )

        changed_rate = np.mean(
            standard_predictions[target]
            != da_predictions[target]
        )

        row = {
            "scenario":
                scenario_name,

            "known_targets":
                ", ".join(
                    sorted(known_targets)
                )
                if known_targets
                else "NONE",

            "evaluated_target":
                target,

            "standard_pr_auc":
                standard["pr_auc"],

            "diagnosis_aware_pr_auc":
                diagnosis_aware["pr_auc"],

            "delta_pr_auc":
                diagnosis_aware["pr_auc"]
                - standard["pr_auc"],

            "standard_roc_auc":
                standard["roc_auc"],

            "diagnosis_aware_roc_auc":
                diagnosis_aware["roc_auc"],

            "delta_roc_auc":
                diagnosis_aware["roc_auc"]
                - standard["roc_auc"],

            "standard_f1":
                standard["f1"],

            "diagnosis_aware_f1":
                diagnosis_aware["f1"],

            "delta_f1":
                diagnosis_aware["f1"]
                - standard["f1"],

            "standard_recall":
                standard["recall"],

            "diagnosis_aware_recall":
                diagnosis_aware["recall"],

            "standard_balanced_accuracy":
                standard[
                    "balanced_accuracy"
                ],

            "diagnosis_aware_balanced_accuracy":
                diagnosis_aware[
                    "balanced_accuracy"
                ],

            "prediction_change_rate":
                changed_rate,
        }

        detail_rows.append(row)
        scenario_details.append(row)

    scenario_df = pd.DataFrame(
        scenario_details
    )

    summary = {
        "scenario":
            scenario_name,

        "known_targets":
            ", ".join(
                sorted(known_targets)
            )
            if known_targets
            else "NONE",

        "unknown_target_count":
            len(unknown_targets),

        "standard_macro_pr_auc":
            scenario_df[
                "standard_pr_auc"
            ].mean(),

        "diagnosis_aware_macro_pr_auc":
            scenario_df[
                "diagnosis_aware_pr_auc"
            ].mean(),

        "delta_macro_pr_auc":
            scenario_df[
                "delta_pr_auc"
            ].mean(),

        "standard_macro_roc_auc":
            scenario_df[
                "standard_roc_auc"
            ].mean(),

        "diagnosis_aware_macro_roc_auc":
            scenario_df[
                "diagnosis_aware_roc_auc"
            ].mean(),

        "delta_macro_roc_auc":
            scenario_df[
                "delta_roc_auc"
            ].mean(),

        "standard_macro_f1":
            scenario_df[
                "standard_f1"
            ].mean(),

        "diagnosis_aware_macro_f1":
            scenario_df[
                "diagnosis_aware_f1"
            ].mean(),

        "delta_macro_f1":
            scenario_df[
                "delta_f1"
            ].mean(),

        "mean_prediction_change_rate":
            scenario_df[
                "prediction_change_rate"
            ].mean(),
    }

    summary_rows.append(summary)


# ============================================================
# BASELINE: NO KNOWN DIAGNOSIS
# ============================================================

evaluate_scenario(
    "0 known diagnoses",
    [],
)


# ============================================================
# ONE KNOWN DIAGNOSIS
# ============================================================

for target in TARGETS:

    evaluate_scenario(
        f"1 known: {target}",
        [target],
    )


# ============================================================
# TWO KNOWN DIAGNOSES
# ============================================================

for known_pair in combinations(
    TARGETS,
    2
):

    evaluate_scenario(
        "2 known: "
        + " + ".join(known_pair),
        known_pair,
    )


# ============================================================
# RESULTS
# ============================================================

detail_df = pd.DataFrame(
    detail_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


print("\n" + "=" * 82)
print("DIAGNOSIS-AWARE SCENARIO SUMMARY")
print("=" * 82)

display_columns = [
    "scenario",
    "known_targets",
    "unknown_target_count",
    "standard_macro_pr_auc",
    "diagnosis_aware_macro_pr_auc",
    "delta_macro_pr_auc",
    "standard_macro_f1",
    "diagnosis_aware_macro_f1",
    "delta_macro_f1",
    "mean_prediction_change_rate",
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 82)
print("PER-TARGET DIAGNOSIS-AWARE EFFECT")
print("=" * 82)

detail_display = [
    "scenario",
    "evaluated_target",
    "standard_pr_auc",
    "diagnosis_aware_pr_auc",
    "delta_pr_auc",
    "standard_f1",
    "diagnosis_aware_f1",
    "delta_f1",
    "prediction_change_rate",
]

print(
    detail_df[
        detail_display
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

summary_path = (
    RESULTS_DIR /
    "diagnosis_aware_summary.csv"
)

detail_path = (
    RESULTS_DIR /
    "diagnosis_aware_per_target.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

detail_df.to_csv(
    detail_path,
    index=False
)


print("\nSaved:")
print(summary_path)
print(detail_path)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

print("\n" + "=" * 82)
print("CHAIN-ORDER INTERPRETATION")
print("=" * 82)

print(
    f"Current winning order: "
    f"{' -> '.join(order)}"
)

print(
    "\nA known diagnosis can influence only "
    "labels appearing AFTER it in the chain."
)

print(
    "Therefore, a known final-chain diagnosis "
    "may correctly produce zero improvement "
    "for earlier unresolved labels."
)

print(
    "\nThe experiment evaluates ONLY labels "
    "that remain unknown."
)

print(
    "Known diagnosis labels are never counted "
    "as successful predictions."
)


print("\nIMPORTANT:")
print(
    "TEST dataset was NOT loaded or used."
)


print("\n" + "=" * 82)
print("STAGE 3.2 DIAGNOSIS-AWARE EXPERIMENT COMPLETE.")
print("=" * 82)