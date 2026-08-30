from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

VAL_PATH = Path(
    "data/processed/validation.csv"
)

BASE_MODEL_PATH = Path(
    "models/federated_classifier_chain.joblib"
)

FAIR_MODEL_PATH = Path(
    "models/fair_federated_classifier_chain.joblib"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GROUPS = {
    "Male": 1,
    "Female": 2,
}


# ============================================================
# LOAD
# ============================================================

print("=" * 96)
print("FEDFAIRCHAIN - STAGE 5.3 FAIRNESS BEFORE VS AFTER COMPARISON")
print("=" * 96)

val_df = pd.read_csv(
    VAL_PATH
)

base_bundle = joblib.load(
    BASE_MODEL_PATH
)

fair_bundle = joblib.load(
    FAIR_MODEL_PATH
)

TARGETS = base_bundle["targets"]


print(
    f"\nValidation patients : "
    f"{len(val_df)}"
)

print(
    "Standard chain      : "
    + " -> ".join(
        base_bundle["order"]
    )
)

print(
    "Fairness-aware chain: "
    + " -> ".join(
        fair_bundle["order"]
    )
)

if (
    list(base_bundle["order"])
    != list(fair_bundle["order"])
):
    raise ValueError(
        "Model chain orders do not match."
    )

if (
    base_bundle["features"]
    != fair_bundle["features"]
):
    raise ValueError(
        "Model feature sets do not match."
    )

print(
    "\n[PASS] Both models use identical "
    "features and chain order."
)

print(
    "[PASS] TEST dataset is NOT loaded."
)


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_model(bundle):

    features = bundle["features"]
    order = list(
        bundle["order"]
    )

    preprocessor = (
        bundle["preprocessor"]
    )

    models = bundle["models"]
    thresholds = bundle["thresholds"]

    X_current = (
        preprocessor.transform(
            val_df[features]
        )
    )

    predictions = {}

    for target in order:

        probability = (
            models[target]
            .predict_proba(
                X_current
            )[:, 1]
        )

        prediction = (
            probability
            >= thresholds[target]
        ).astype(int)

        predictions[target] = (
            prediction
        )

        X_current = np.column_stack(
            [
                X_current,
                prediction,
            ]
        )

    return predictions


base_predictions = run_model(
    base_bundle
)

fair_predictions = run_model(
    fair_bundle
)


# ============================================================
# GROUP METRICS
# ============================================================

def group_metrics(
    y_true,
    y_pred,
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

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

    positive_rate = (
        y_pred.mean()
    )

    if (fp + tn) > 0:

        fpr = (
            fp / (fp + tn)
        )

    else:

        fpr = np.nan


    return {
        "actual_prevalence":
            y_true.mean(),

        "predicted_positive_rate":
            positive_rate,

        "recall":
            recall,

        "false_positive_rate":
            fpr,

        "precision":
            precision,

        "f1":
            f1,
    }


# ============================================================
# FAIRNESS GAP FUNCTION
# ============================================================

def calculate_fairness(
    predictions,
    target,
):

    metrics = {}

    for group_name, sex_code in (
        GROUPS.items()
    ):

        mask = (
            val_df["SEX"]
            == sex_code
        ).to_numpy()

        y_true = (
            val_df.loc[
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

        metrics[group_name] = (
            group_metrics(
                y_true,
                y_pred,
            )
        )


    male = metrics["Male"]
    female = metrics["Female"]


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


    return {
        "groups":
            metrics,

        "dp":
            dp,

        "eo":
            eo,

        "fpr_gap":
            fpr_gap,
    }


# ============================================================
# COMPARE
# ============================================================

comparison_rows = []
group_rows = []


for target in TARGETS:

    before = calculate_fairness(
        base_predictions,
        target,
    )

    after = calculate_fairness(
        fair_predictions,
        target,
    )


    # --------------------------------------------------------
    # Group-level details
    # --------------------------------------------------------

    for group in [
        "Male",
        "Female",
    ]:

        before_group = (
            before["groups"][group]
        )

        after_group = (
            after["groups"][group]
        )

        group_rows.append(
            {
                "target":
                    target,

                "group":
                    group,

                "before_recall":
                    before_group[
                        "recall"
                    ],

                "after_recall":
                    after_group[
                        "recall"
                    ],

                "before_fpr":
                    before_group[
                        "false_positive_rate"
                    ],

                "after_fpr":
                    after_group[
                        "false_positive_rate"
                    ],

                "before_precision":
                    before_group[
                        "precision"
                    ],

                "after_precision":
                    after_group[
                        "precision"
                    ],

                "before_f1":
                    before_group[
                        "f1"
                    ],

                "after_f1":
                    after_group[
                        "f1"
                    ],

                "before_positive_rate":
                    before_group[
                        "predicted_positive_rate"
                    ],

                "after_positive_rate":
                    after_group[
                        "predicted_positive_rate"
                    ],
            }
        )


    # --------------------------------------------------------
    # Fairness gap comparison
    # --------------------------------------------------------

    before_abs_dp = abs(
        before["dp"]
    )

    after_abs_dp = abs(
        after["dp"]
    )

    before_abs_eo = abs(
        before["eo"]
    )

    after_abs_eo = abs(
        after["eo"]
    )

    before_abs_fpr = abs(
        before["fpr_gap"]
    )

    after_abs_fpr = abs(
        after["fpr_gap"]
    )


    comparison_rows.append(
        {
            "target":
                target,

            "before_dp":
                before["dp"],

            "after_dp":
                after["dp"],

            "before_abs_dp":
                before_abs_dp,

            "after_abs_dp":
                after_abs_dp,

            "dp_reduction":
                before_abs_dp
                - after_abs_dp,

            "before_eo":
                before["eo"],

            "after_eo":
                after["eo"],

            "before_abs_eo":
                before_abs_eo,

            "after_abs_eo":
                after_abs_eo,

            "eo_reduction":
                before_abs_eo
                - after_abs_eo,

            "before_fpr_gap":
                before["fpr_gap"],

            "after_fpr_gap":
                after["fpr_gap"],

            "before_abs_fpr_gap":
                before_abs_fpr,

            "after_abs_fpr_gap":
                after_abs_fpr,

            "fpr_gap_reduction":
                before_abs_fpr
                - after_abs_fpr,
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

group_df = pd.DataFrame(
    group_rows
)


# ============================================================
# PRINT PER-DISEASE COMPARISON
# ============================================================

print("\n" + "=" * 96)
print("FAIRNESS GAP COMPARISON")
print("=" * 96)


for _, row in comparison_df.iterrows():

    print(
        f"\n{row['target']}"
    )

    print(
        "  Demographic Parity:"
    )

    print(
        f"    Before absolute gap = "
        f"{row['before_abs_dp']:.4f}"
    )

    print(
        f"    After absolute gap  = "
        f"{row['after_abs_dp']:.4f}"
    )

    print(
        f"    Reduction           = "
        f"{row['dp_reduction']:+.4f}"
    )


    print(
        "  Equal Opportunity:"
    )

    print(
        f"    Before absolute gap = "
        f"{row['before_abs_eo']:.4f}"
    )

    print(
        f"    After absolute gap  = "
        f"{row['after_abs_eo']:.4f}"
    )

    print(
        f"    Reduction           = "
        f"{row['eo_reduction']:+.4f}"
    )


    print(
        "  False Positive Rate:"
    )

    print(
        f"    Before absolute gap = "
        f"{row['before_abs_fpr_gap']:.4f}"
    )

    print(
        f"    After absolute gap  = "
        f"{row['after_abs_fpr_gap']:.4f}"
    )

    print(
        f"    Reduction           = "
        f"{row['fpr_gap_reduction']:+.4f}"
    )


# ============================================================
# PARTICULAR GROUP RECALL
# ============================================================

print("\n" + "=" * 96)
print("GROUP RECALL BEFORE VS AFTER")
print("=" * 96)

recall_display = group_df[
    [
        "target",
        "group",
        "before_recall",
        "after_recall",
        "before_f1",
        "after_f1",
    ]
]

print(
    recall_display.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# OVERALL FAIRNESS SUMMARY
# ============================================================

before_mean_dp = (
    comparison_df[
        "before_abs_dp"
    ].mean()
)

after_mean_dp = (
    comparison_df[
        "after_abs_dp"
    ].mean()
)


before_mean_eo = (
    comparison_df[
        "before_abs_eo"
    ].mean()
)

after_mean_eo = (
    comparison_df[
        "after_abs_eo"
    ].mean()
)


before_mean_fpr = (
    comparison_df[
        "before_abs_fpr_gap"
    ].mean()
)

after_mean_fpr = (
    comparison_df[
        "after_abs_fpr_gap"
    ].mean()
)


print("\n" + "=" * 96)
print("MEAN ABSOLUTE FAIRNESS GAPS")
print("=" * 96)

print(
    "\nDemographic Parity"
)

print(
    f"Before : {before_mean_dp:.4f}"
)

print(
    f"After  : {after_mean_dp:.4f}"
)

print(
    f"Change : "
    f"{before_mean_dp - after_mean_dp:+.4f}"
)


print(
    "\nEqual Opportunity"
)

print(
    f"Before : {before_mean_eo:.4f}"
)

print(
    f"After  : {after_mean_eo:.4f}"
)

print(
    f"Change : "
    f"{before_mean_eo - after_mean_eo:+.4f}"
)


print(
    "\nFalse Positive Rate Gap"
)

print(
    f"Before : {before_mean_fpr:.4f}"
)

print(
    f"After  : {after_mean_fpr:.4f}"
)

print(
    f"Change : "
    f"{before_mean_fpr - after_mean_fpr:+.4f}"
)


# ============================================================
# SAVE
# ============================================================

comparison_path = (
    RESULTS_DIR /
    "fairness_before_after_comparison.csv"
)

group_path = (
    RESULTS_DIR /
    "fairness_before_after_group_metrics.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)

group_df.to_csv(
    group_path,
    index=False,
)


print("\nSaved:")
print(comparison_path)
print(group_path)


print("\nIMPORTANT:")

print(
    "Positive reduction values mean "
    "the absolute fairness gap became smaller."
)

print(
    "Negative reduction values mean "
    "that particular fairness gap became larger."
)

print(
    "Fairness must be interpreted together "
    "with predictive performance."
)

print(
    "TEST dataset was NOT loaded or used."
)


print("\n" + "=" * 96)
print("STAGE 5.3 FAIRNESS COMPARISON COMPLETE.")
print("=" * 96)