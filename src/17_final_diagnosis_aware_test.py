from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PATH = Path(
    "data/processed/test.csv"
)

MODEL_PATH = Path(
    "models/fair_federated_classifier_chain.joblib"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD FINAL FROZEN MODEL + TEST SET
# ============================================================

print("=" * 96)
print("FEDFAIRCHAIN - STAGE 8.2 FINAL DIAGNOSIS-AWARE TEST")
print("=" * 96)

test_df = pd.read_csv(
    TEST_PATH
)

bundle = joblib.load(
    MODEL_PATH
)

FEATURES = list(
    bundle["features"]
)

ORDER = list(
    bundle["order"]
)

THRESHOLDS = bundle["thresholds"]
PREPROCESSOR = bundle["preprocessor"]
MODELS = bundle["models"]


print(
    f"\nTest patients     : {len(test_df)}"
)

print(
    "Frozen chain      : "
    + " -> ".join(ORDER)
)

print(
    "[PASS] No training or threshold tuning."
)

print(
    "[PASS] Known diagnoses are excluded "
    "from scoring."
)


# ============================================================
# BASE FEATURES
# ============================================================

X_BASE = PREPROCESSOR.transform(
    test_df[FEATURES]
)


# ============================================================
# STANDARD CHAIN
# ============================================================

def run_standard_chain():

    probabilities = {}
    predictions = {}

    X_current = X_BASE.copy()

    for target in ORDER:

        probability = (
            MODELS[target]
            .predict_proba(
                X_current
            )[:, 1]
        )

        prediction = (
            probability
            >= THRESHOLDS[target]
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


standard_probabilities, standard_predictions = (
    run_standard_chain()
)


# ============================================================
# DIAGNOSIS-AWARE CHAIN
# ============================================================

def run_diagnosis_aware_chain(
    known_targets,
):

    probabilities = {}
    predictions = {}
    chain_values = {}

    X_current = X_BASE.copy()

    for target in ORDER:

        probability = (
            MODELS[target]
            .predict_proba(
                X_current
            )[:, 1]
        )

        model_prediction = (
            probability
            >= THRESHOLDS[target]
        ).astype(int)


        # ----------------------------------------------------
        # KNOWN TARGET
        #
        # Use actual known diagnosis only as chain context.
        # Do not score this target.
        # ----------------------------------------------------

        if target in known_targets:

            value_for_chain = (
                test_df[target]
                .astype(int)
                .to_numpy()
            )

        else:

            value_for_chain = (
                model_prediction
            )

            probabilities[target] = (
                probability
            )

            predictions[target] = (
                model_prediction
            )


        chain_values[target] = (
            value_for_chain
        )


        X_current = np.column_stack(
            [
                X_current,
                value_for_chain,
            ]
        )


    return (
        probabilities,
        predictions,
        chain_values,
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def calculate_macro_metrics(
    probabilities,
    predictions,
    evaluated_targets,
):

    pr_scores = []
    roc_scores = []
    f1_scores = []


    for target in evaluated_targets:

        y_true = (
            test_df[target]
            .astype(int)
            .to_numpy()
        )

        pr_scores.append(
            average_precision_score(
                y_true,
                probabilities[target],
            )
        )

        roc_scores.append(
            roc_auc_score(
                y_true,
                probabilities[target],
            )
        )

        f1_scores.append(
            f1_score(
                y_true,
                predictions[target],
                zero_division=0,
            )
        )


    return {
        "macro_pr_auc":
            float(
                np.mean(pr_scores)
            ),

        "macro_roc_auc":
            float(
                np.mean(roc_scores)
            ),

        "macro_f1":
            float(
                np.mean(f1_scores)
            ),
    }


# ============================================================
# DIAGNOSIS-AWARE SCENARIOS
# ============================================================

SCENARIOS = [

    (
        "Known Diabetes",
        {"DIABETES"},
    ),

    (
        "Known CVD",
        {"CVD"},
    ),

    (
        "Known Hypertension",
        {"HYPERTENSION"},
    ),

    (
        "Known Diabetes + CVD",
        {"DIABETES", "CVD"},
    ),

    (
        "Known Diabetes + Hypertension",
        {"DIABETES", "HYPERTENSION"},
    ),

    (
        "Known CVD + Hypertension",
        {"CVD", "HYPERTENSION"},
    ),
]


summary_rows = []
per_target_rows = []


print("\n" + "=" * 96)
print("FINAL DIAGNOSIS-AWARE SCENARIOS")
print("=" * 96)


for scenario_name, known_targets in SCENARIOS:

    evaluated_targets = [
        target
        for target in ORDER
        if target not in known_targets
    ]


    da_probabilities, da_predictions, _ = (
        run_diagnosis_aware_chain(
            known_targets
        )
    )


    # --------------------------------------------------------
    # STANDARD MODEL METRICS
    #
    # Evaluate only labels that remain unknown
    # in this scenario.
    # --------------------------------------------------------

    standard_probs_subset = {
        target:
            standard_probabilities[target]

        for target in evaluated_targets
    }

    standard_preds_subset = {
        target:
            standard_predictions[target]

        for target in evaluated_targets
    }


    standard_metrics = (
        calculate_macro_metrics(
            standard_probs_subset,
            standard_preds_subset,
            evaluated_targets,
        )
    )


    da_metrics = (
        calculate_macro_metrics(
            da_probabilities,
            da_predictions,
            evaluated_targets,
        )
    )


    # --------------------------------------------------------
    # PREDICTION CHANGE RATE
    # --------------------------------------------------------

    standard_matrix = np.column_stack(
        [
            standard_predictions[
                target
            ]
            for target
            in evaluated_targets
        ]
    )

    da_matrix = np.column_stack(
        [
            da_predictions[
                target
            ]
            for target
            in evaluated_targets
        ]
    )


    change_rate = float(
        np.mean(
            standard_matrix
            != da_matrix
        )
    )


    summary_rows.append(
        {
            "scenario":
                scenario_name,

            "known_targets":
                " + ".join(
                    sorted(
                        known_targets
                    )
                ),

            "evaluated_targets":
                " + ".join(
                    evaluated_targets
                ),

            "standard_macro_pr_auc":
                standard_metrics[
                    "macro_pr_auc"
                ],

            "diagnosis_aware_macro_pr_auc":
                da_metrics[
                    "macro_pr_auc"
                ],

            "pr_auc_delta":
                da_metrics[
                    "macro_pr_auc"
                ]
                -
                standard_metrics[
                    "macro_pr_auc"
                ],

            "standard_macro_roc_auc":
                standard_metrics[
                    "macro_roc_auc"
                ],

            "diagnosis_aware_macro_roc_auc":
                da_metrics[
                    "macro_roc_auc"
                ],

            "roc_auc_delta":
                da_metrics[
                    "macro_roc_auc"
                ]
                -
                standard_metrics[
                    "macro_roc_auc"
                ],

            "standard_macro_f1":
                standard_metrics[
                    "macro_f1"
                ],

            "diagnosis_aware_macro_f1":
                da_metrics[
                    "macro_f1"
                ],

            "f1_delta":
                da_metrics[
                    "macro_f1"
                ]
                -
                standard_metrics[
                    "macro_f1"
                ],

            "prediction_change_rate":
                change_rate,
        }
    )


    # --------------------------------------------------------
    # PER-TARGET RESULTS
    # --------------------------------------------------------

    for target in evaluated_targets:

        y_true = (
            test_df[target]
            .astype(int)
            .to_numpy()
        )


        standard_pr = (
            average_precision_score(
                y_true,
                standard_probabilities[
                    target
                ],
            )
        )

        da_pr = (
            average_precision_score(
                y_true,
                da_probabilities[
                    target
                ],
            )
        )


        standard_f1 = (
            f1_score(
                y_true,
                standard_predictions[
                    target
                ],
                zero_division=0,
            )
        )

        da_f1 = (
            f1_score(
                y_true,
                da_predictions[
                    target
                ],
                zero_division=0,
            )
        )


        target_change_rate = float(
            np.mean(
                standard_predictions[
                    target
                ]
                !=
                da_predictions[
                    target
                ]
            )
        )


        per_target_rows.append(
            {
                "scenario":
                    scenario_name,

                "target":
                    target,

                "standard_pr_auc":
                    standard_pr,

                "diagnosis_aware_pr_auc":
                    da_pr,

                "pr_auc_delta":
                    da_pr
                    - standard_pr,

                "standard_f1":
                    standard_f1,

                "diagnosis_aware_f1":
                    da_f1,

                "f1_delta":
                    da_f1
                    - standard_f1,

                "prediction_change_rate":
                    target_change_rate,
            }
        )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\n{scenario_name}"
    )

    print(
        "Known      : "
        + " + ".join(
            sorted(
                known_targets
            )
        )
    )

    print(
        "Evaluating : "
        + " + ".join(
            evaluated_targets
        )
    )

    print(
        f"Macro PR-AUC: "
        f"{standard_metrics['macro_pr_auc']:.4f}"
        f" -> "
        f"{da_metrics['macro_pr_auc']:.4f} "
        f"("
        f"{da_metrics['macro_pr_auc'] - standard_metrics['macro_pr_auc']:+.4f}"
        f")"
    )

    print(
        f"Macro ROC-AUC: "
        f"{standard_metrics['macro_roc_auc']:.4f}"
        f" -> "
        f"{da_metrics['macro_roc_auc']:.4f} "
        f"("
        f"{da_metrics['macro_roc_auc'] - standard_metrics['macro_roc_auc']:+.4f}"
        f")"
    )

    print(
        f"Macro F1: "
        f"{standard_metrics['macro_f1']:.4f}"
        f" -> "
        f"{da_metrics['macro_f1']:.4f} "
        f"("
        f"{da_metrics['macro_f1'] - standard_metrics['macro_f1']:+.4f}"
        f")"
    )

    print(
        f"Prediction change rate: "
        f"{change_rate:.4f}"
    )


# ============================================================
# CREATE DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

per_target_df = pd.DataFrame(
    per_target_rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 96)
print("FINAL DIAGNOSIS-AWARE SUMMARY")
print("=" * 96)


display_columns = [
    "scenario",
    "standard_macro_pr_auc",
    "diagnosis_aware_macro_pr_auc",
    "pr_auc_delta",
    "standard_macro_f1",
    "diagnosis_aware_macro_f1",
    "f1_delta",
    "prediction_change_rate",
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 96)
print("FINAL PER-TARGET DIAGNOSIS-AWARE EFFECTS")
print("=" * 96)


print(
    per_target_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SAVE
# ============================================================

summary_path = (
    RESULTS_DIR /
    "final_test_diagnosis_aware_summary.csv"
)

target_path = (
    RESULTS_DIR /
    "final_test_diagnosis_aware_per_target.csv"
)


summary_df.to_csv(
    summary_path,
    index=False,
)

per_target_df.to_csv(
    target_path,
    index=False,
)


print("\nSaved:")
print(summary_path)
print(target_path)


print("\nIMPORTANT:")
print(
    "Known diagnoses were used only as "
    "classifier-chain context."
)

print(
    "Known diagnoses were excluded from scoring."
)

print(
    "No model parameter, threshold, chain order, "
    "or preprocessing component was changed."
)


print("\n" + "=" * 96)
print("STAGE 8.2 FINAL DIAGNOSIS-AWARE TEST COMPLETE")
print("=" * 96)