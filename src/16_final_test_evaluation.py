from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PATH = Path(
    "data/processed/test.csv"
)

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Canonical output order for comparison.
TARGETS = [
    "DIABETES",
    "HYPERTENSION",
    "CVD",
]


MODEL_NAMES = {
    "M1": "Independent Centralized",
    "M2": "Centralized Classifier Chain",
    "M3": "Federated Classifier Chain",
    "M4": "FedFairChain",
}


# ============================================================
# IMPORTANT FINAL-TEST NOTICE
# ============================================================

print("=" * 100)
print("FEDFAIRCHAIN - STAGE 8 FINAL HELD-OUT TEST EVALUATION")
print("=" * 100)

print(
    "\nIMPORTANT:"
)

print(
    "The held-out TEST set is being used for final "
    "evaluation only."
)

print(
    "No fitting, retraining, threshold tuning, chain-order "
    "selection, or federated-round selection is performed."
)


# ============================================================
# LOAD TEST SET
# ============================================================

test_df = pd.read_csv(
    TEST_PATH
)

print(
    f"\nTest patients: {len(test_df)}"
)


required_columns = {
    "SEQN",
    "SEX",
    *TARGETS,
}

missing_columns = (
    required_columns
    - set(test_df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required test columns: "
        f"{sorted(missing_columns)}"
    )


for target in TARGETS:

    if test_df[target].isna().any():
        raise ValueError(
            f"Missing labels found in "
            f"test target {target}."
        )

    unique_values = set(
        test_df[target]
        .astype(int)
        .unique()
    )

    if not unique_values.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"Non-binary values found "
            f"in {target}."
        )


print(
    "[PASS] Final test labels are valid."
)

print(
    "[PASS] No model fitting will occur "
    "inside this script."
)


# ============================================================
# LOAD ALL FROZEN MODELS
# ============================================================

print("\n" + "=" * 100)
print("LOADING FROZEN MODELS")
print("=" * 100)


# ------------------------------------------------------------
# M1 - Independent centralized classifiers
# ------------------------------------------------------------

independent_models = {
    "DIABETES":
        joblib.load(
            MODEL_DIR /
            "independent_diabetes.joblib"
        ),

    "HYPERTENSION":
        joblib.load(
            MODEL_DIR /
            "independent_hypertension.joblib"
        ),

    "CVD":
        joblib.load(
            MODEL_DIR /
            "independent_cvd.joblib"
        ),
}


with open(
    MODEL_DIR /
    "independent_thresholds.json",
    "r",
    encoding="utf-8",
) as file:

    independent_thresholds = (
        json.load(file)
    )


# ------------------------------------------------------------
# M2 - Centralized classifier chain
# ------------------------------------------------------------

central_chain = joblib.load(
    MODEL_DIR /
    "best_classifier_chain.joblib"
)


# ------------------------------------------------------------
# M3 - Federated classifier chain
# ------------------------------------------------------------

federated_chain = joblib.load(
    MODEL_DIR /
    "federated_classifier_chain.joblib"
)


# ------------------------------------------------------------
# M4 - Fairness-aware federated classifier chain
# ------------------------------------------------------------

fair_chain = joblib.load(
    MODEL_DIR /
    "fair_federated_classifier_chain.joblib"
)


print(
    "M1 loaded: Independent Centralized"
)

print(
    "M2 loaded: Centralized Classifier Chain"
)

print(
    "M3 loaded: Federated Classifier Chain"
)

print(
    "M4 loaded: FedFairChain"
)


# ============================================================
# CHECK MODEL CONTRACTS
# ============================================================

chain_bundles = {
    "M2": central_chain,
    "M3": federated_chain,
    "M4": fair_chain,
}


reference_features = list(
    fair_chain["features"]
)


if "SEX" in reference_features:
    raise RuntimeError(
        "SEX must not be present "
        "in prediction features."
    )


for model_id, bundle in (
    chain_bundles.items()
):

    if list(
        bundle["features"]
    ) != reference_features:

        raise RuntimeError(
            f"{model_id} feature contract "
            f"does not match."
        )


print(
    "\n[PASS] M2, M3 and M4 use the "
    "same base feature contract."
)

print(
    f"[PASS] Number of prediction features: "
    f"{len(reference_features)}"
)

print(
    "[PASS] SEX is excluded from prediction inputs."
)


print("\nFrozen chain orders:")

for model_id, bundle in (
    chain_bundles.items()
):

    print(
        f"{model_id}: "
        + " -> ".join(
            bundle["order"]
        )
    )


# ============================================================
# M1 INFERENCE
# ============================================================

def predict_independent():

    probabilities = {}
    predictions = {}


    for target in TARGETS:

        model = (
            independent_models[
                target
            ]
        )

        probability = (
            model.predict_proba(
                test_df[
                    reference_features
                ]
            )[:, 1]
        )

        threshold = float(
            independent_thresholds[
                target
            ]
        )

        prediction = (
            probability
            >= threshold
        ).astype(int)


        probabilities[target] = (
            probability
        )

        predictions[target] = (
            prediction
        )


    return (
        probabilities,
        predictions,
    )


# ============================================================
# CLASSIFIER-CHAIN INFERENCE
# ============================================================

def predict_chain(bundle):

    features = list(
        bundle["features"]
    )

    order = list(
        bundle["order"]
    )

    thresholds = (
        bundle["thresholds"]
    )

    preprocessor = (
        bundle["preprocessor"]
    )

    models = (
        bundle["models"]
    )


    # IMPORTANT:
    # The already-fitted preprocessor is reused.
    # There is NO fit() call here.

    X_current = (
        preprocessor.transform(
            test_df[features]
        )
    )


    probabilities = {}
    predictions = {}


    for target in order:

        probability = (
            models[target]
            .predict_proba(
                X_current
            )[:, 1]
        )

        threshold = float(
            thresholds[target]
        )

        prediction = (
            probability
            >= threshold
        ).astype(int)


        probabilities[target] = (
            probability
        )

        predictions[target] = (
            prediction
        )


        # Pass frozen binary chain output
        # to the next classifier.

        X_current = np.column_stack(
            [
                X_current,
                prediction,
            ]
        )


    return (
        probabilities,
        predictions,
    )


# ============================================================
# RUN ALL FOUR MODELS ON TEST SET
# ============================================================

print("\n" + "=" * 100)
print("RUNNING FINAL TEST INFERENCE")
print("=" * 100)


all_outputs = {}


all_outputs["M1"] = (
    predict_independent()
)

print(
    "[PASS] M1 inference complete."
)


all_outputs["M2"] = (
    predict_chain(
        central_chain
    )
)

print(
    "[PASS] M2 inference complete."
)


all_outputs["M3"] = (
    predict_chain(
        federated_chain
    )
)

print(
    "[PASS] M3 inference complete."
)


all_outputs["M4"] = (
    predict_chain(
        fair_chain
    )
)

print(
    "[PASS] M4 inference complete."
)


# ============================================================
# PER-LABEL METRICS
# ============================================================

per_label_rows = []


for model_id, (
    probabilities,
    predictions,
) in all_outputs.items():


    for target in TARGETS:

        y_true = (
            test_df[target]
            .astype(int)
            .to_numpy()
        )

        probability = (
            probabilities[target]
        )

        prediction = (
            predictions[target]
        )


        per_label_rows.append(
            {
                "model_id":
                    model_id,

                "model":
                    MODEL_NAMES[
                        model_id
                    ],

                "target":
                    target,

                "roc_auc":
                    roc_auc_score(
                        y_true,
                        probability,
                    ),

                "pr_auc":
                    average_precision_score(
                        y_true,
                        probability,
                    ),

                "precision":
                    precision_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    ),

                "recall":
                    recall_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    ),

                "f1":
                    f1_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    ),

                "balanced_accuracy":
                    balanced_accuracy_score(
                        y_true,
                        prediction,
                    ),
            }
        )


per_label_df = pd.DataFrame(
    per_label_rows
)


# ============================================================
# MULTI-LABEL / MODEL-LEVEL METRICS
# ============================================================

summary_rows = []


y_true_multi = (
    test_df[TARGETS]
    .astype(int)
    .to_numpy()
)


for model_id, (
    probabilities,
    predictions,
) in all_outputs.items():


    model_metrics = (
        per_label_df[
            per_label_df[
                "model_id"
            ] == model_id
        ]
    )


    y_pred_multi = np.column_stack(
        [
            predictions[target]
            for target in TARGETS
        ]
    )


    summary_rows.append(
        {
            "model_id":
                model_id,

            "model":
                MODEL_NAMES[
                    model_id
                ],

            "macro_roc_auc":
                model_metrics[
                    "roc_auc"
                ].mean(),

            "macro_pr_auc":
                model_metrics[
                    "pr_auc"
                ].mean(),

            "macro_precision":
                model_metrics[
                    "precision"
                ].mean(),

            "macro_recall":
                model_metrics[
                    "recall"
                ].mean(),

            "macro_f1":
                model_metrics[
                    "f1"
                ].mean(),

            "macro_balanced_accuracy":
                model_metrics[
                    "balanced_accuracy"
                ].mean(),

            "micro_f1":
                f1_score(
                    y_true_multi,
                    y_pred_multi,
                    average="micro",
                    zero_division=0,
                ),

            "hamming_loss":
                hamming_loss(
                    y_true_multi,
                    y_pred_multi,
                ),

            "exact_match_accuracy":
                accuracy_score(
                    y_true_multi,
                    y_pred_multi,
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT PER-LABEL RESULTS
# ============================================================

print("\n" + "=" * 100)
print("FINAL HELD-OUT TEST RESULTS - PER DISEASE")
print("=" * 100)


for model_id in [
    "M1",
    "M2",
    "M3",
    "M4",
]:

    print(
        f"\n{model_id} - "
        f"{MODEL_NAMES[model_id]}"
    )

    display_df = (
        per_label_df[
            per_label_df[
                "model_id"
            ] == model_id
        ][
            [
                "target",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
            ]
        ]
    )


    print(
        display_df.to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:.4f}"
            ),
        )
    )


# ============================================================
# PRINT OVERALL COMPARISON
# ============================================================

print("\n" + "=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)


display_summary = (
    summary_df[
        [
            "model_id",
            "model",
            "macro_roc_auc",
            "macro_pr_auc",
            "macro_f1",
            "micro_f1",
            "macro_recall",
            "macro_balanced_accuracy",
            "hamming_loss",
            "exact_match_accuracy",
        ]
    ]
)


print(
    display_summary.to_string(
        index=False,
        float_format=(
            lambda x:
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# FAIRNESS METRICS
# ============================================================

def group_metrics(
    y_true,
    y_pred,
):

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()
    )


    predicted_positive_rate = float(
        np.mean(
            y_pred
        )
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    if (
        fp + tn
    ) > 0:

        false_positive_rate = (
            fp / (
                fp + tn
            )
        )

    else:

        false_positive_rate = (
            np.nan
        )


    return {
        "n":
            len(y_true),

        "prevalence":
            float(
                np.mean(
                    y_true
                )
            ),

        "predicted_positive_rate":
            predicted_positive_rate,

        "recall":
            recall,

        "false_positive_rate":
            false_positive_rate,

        "precision":
            precision,

        "f1":
            f1,
    }


GROUPS = {
    "Male": 1,
    "Female": 2,
}


fairness_group_rows = []
fairness_gap_rows = []


for model_id, (
    probabilities,
    predictions,
) in all_outputs.items():


    for target in TARGETS:

        group_results = {}


        for group_name, sex_code in (
            GROUPS.items()
        ):

            mask = (
                test_df["SEX"]
                == sex_code
            ).to_numpy()


            y_true = (
                test_df.loc[
                    mask,
                    target
                ]
                .astype(int)
                .to_numpy()
            )


            y_pred = (
                predictions[target][
                    mask
                ]
            )


            metrics = (
                group_metrics(
                    y_true,
                    y_pred,
                )
            )


            group_results[
                group_name
            ] = metrics


            fairness_group_rows.append(
                {
                    "model_id":
                        model_id,

                    "model":
                        MODEL_NAMES[
                            model_id
                        ],

                    "target":
                        target,

                    "group":
                        group_name,

                    **metrics,
                }
            )


        male = (
            group_results[
                "Male"
            ]
        )

        female = (
            group_results[
                "Female"
            ]
        )


        dp = (
            female[
                "predicted_positive_rate"
            ]
            -
            male[
                "predicted_positive_rate"
            ]
        )


        eo = (
            female["recall"]
            -
            male["recall"]
        )


        fpr_gap = (
            female[
                "false_positive_rate"
            ]
            -
            male[
                "false_positive_rate"
            ]
        )


        fairness_gap_rows.append(
            {
                "model_id":
                    model_id,

                "model":
                    MODEL_NAMES[
                        model_id
                    ],

                "target":
                    target,

                "demographic_parity_difference":
                    dp,

                "abs_demographic_parity_difference":
                    abs(dp),

                "equal_opportunity_difference":
                    eo,

                "abs_equal_opportunity_difference":
                    abs(eo),

                "false_positive_rate_difference":
                    fpr_gap,

                "abs_false_positive_rate_difference":
                    abs(
                        fpr_gap
                    ),
            }
        )


fairness_group_df = pd.DataFrame(
    fairness_group_rows
)

fairness_gap_df = pd.DataFrame(
    fairness_gap_rows
)


# ============================================================
# FAIRNESS SUMMARY PER MODEL
# ============================================================

fairness_summary_rows = []


for model_id in [
    "M1",
    "M2",
    "M3",
    "M4",
]:

    model_fairness = (
        fairness_gap_df[
            fairness_gap_df[
                "model_id"
            ] == model_id
        ]
    )


    fairness_summary_rows.append(
        {
            "model_id":
                model_id,

            "model":
                MODEL_NAMES[
                    model_id
                ],

            "mean_abs_dp_gap":
                model_fairness[
                    "abs_demographic_parity_difference"
                ].mean(),

            "mean_abs_eo_gap":
                model_fairness[
                    "abs_equal_opportunity_difference"
                ].mean(),

            "mean_abs_fpr_gap":
                model_fairness[
                    "abs_false_positive_rate_difference"
                ].mean(),
        }
    )


fairness_summary_df = pd.DataFrame(
    fairness_summary_rows
)


print("\n" + "=" * 100)
print("FINAL TEST FAIRNESS SUMMARY")
print("=" * 100)


print(
    fairness_summary_df.to_string(
        index=False,
        float_format=(
            lambda x:
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# M3 VS M4 FAIRNESS COMPARISON
# ============================================================

print("\n" + "=" * 100)
print("M3 FEDERATED VS M4 FEDFAIRCHAIN FAIRNESS")
print("=" * 100)


m3_fairness = (
    fairness_gap_df[
        fairness_gap_df[
            "model_id"
        ] == "M3"
    ]
    .set_index(
        "target"
    )
)


m4_fairness = (
    fairness_gap_df[
        fairness_gap_df[
            "model_id"
        ] == "M4"
    ]
    .set_index(
        "target"
    )
)


comparison_rows = []


for target in TARGETS:

    before = (
        m3_fairness.loc[
            target
        ]
    )

    after = (
        m4_fairness.loc[
            target
        ]
    )


    comparison_rows.append(
        {
            "target":
                target,

            "M3_abs_DP":
                before[
                    "abs_demographic_parity_difference"
                ],

            "M4_abs_DP":
                after[
                    "abs_demographic_parity_difference"
                ],

            "DP_reduction":
                before[
                    "abs_demographic_parity_difference"
                ]
                -
                after[
                    "abs_demographic_parity_difference"
                ],

            "M3_abs_EO":
                before[
                    "abs_equal_opportunity_difference"
                ],

            "M4_abs_EO":
                after[
                    "abs_equal_opportunity_difference"
                ],

            "EO_reduction":
                before[
                    "abs_equal_opportunity_difference"
                ]
                -
                after[
                    "abs_equal_opportunity_difference"
                ],

            "M3_abs_FPR":
                before[
                    "abs_false_positive_rate_difference"
                ],

            "M4_abs_FPR":
                after[
                    "abs_false_positive_rate_difference"
                ],

            "FPR_reduction":
                before[
                    "abs_false_positive_rate_difference"
                ]
                -
                after[
                    "abs_false_positive_rate_difference"
                ],
        }
    )


m3_m4_fairness_df = pd.DataFrame(
    comparison_rows
)


print(
    m3_m4_fairness_df.to_string(
        index=False,
        float_format=(
            lambda x:
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

per_label_path = (
    RESULTS_DIR /
    "final_test_per_label_metrics.csv"
)

summary_path = (
    RESULTS_DIR /
    "final_test_model_comparison.csv"
)

fairness_groups_path = (
    RESULTS_DIR /
    "final_test_fairness_group_metrics.csv"
)

fairness_gaps_path = (
    RESULTS_DIR /
    "final_test_fairness_gaps.csv"
)

fairness_summary_path = (
    RESULTS_DIR /
    "final_test_fairness_summary.csv"
)

fairness_comparison_path = (
    RESULTS_DIR /
    "final_test_m3_vs_m4_fairness.csv"
)


per_label_df.to_csv(
    per_label_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)

fairness_group_df.to_csv(
    fairness_groups_path,
    index=False,
)

fairness_gap_df.to_csv(
    fairness_gaps_path,
    index=False,
)

fairness_summary_df.to_csv(
    fairness_summary_path,
    index=False,
)

m3_m4_fairness_df.to_csv(
    fairness_comparison_path,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("FINAL RESULT FILES SAVED")
print("=" * 100)

print(per_label_path)
print(summary_path)
print(fairness_groups_path)
print(fairness_gaps_path)
print(fairness_summary_path)
print(fairness_comparison_path)


print("\n" + "=" * 100)
print("STAGE 8.1 FINAL HELD-OUT TEST EVALUATION COMPLETE")
print("=" * 100)

print(
    "\nThese are FINAL held-out test results."
)

print(
    "Do not tune model parameters or thresholds "
    "based on these results."
)