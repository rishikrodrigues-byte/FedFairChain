from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

VAL_PATH = Path(
    "data/processed/validation.csv"
)

MODEL_PATH = Path(
    "models/federated_classifier_chain.joblib"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA + FEDERATED MODEL
# ============================================================

print("=" * 88)
print("FEDFAIRCHAIN - STAGE 5.1 BASELINE FAIRNESS AUDIT")
print("=" * 88)

val_df = pd.read_csv(
    VAL_PATH
)

bundle = joblib.load(
    MODEL_PATH
)

FEATURES = bundle["features"]
TARGETS = bundle["targets"]
ORDER = list(
    bundle["order"]
)

THRESHOLDS = bundle["thresholds"]
PREPROCESSOR = bundle["preprocessor"]
MODELS = bundle["models"]


print(
    f"\nValidation patients : "
    f"{len(val_df)}"
)

print(
    "Classifier chain   : "
    + " -> ".join(ORDER)
)

print(
    "\nProtected attribute: SEX"
)

print(
    "[PASS] SEX is NOT included "
    "inside model predictors."
)

if "SEX" in FEATURES:
    raise ValueError(
        "SEX unexpectedly found in FEATURES."
    )

print(
    "[PASS] TEST dataset is not loaded."
)


# ============================================================
# CHECK SEX VALUES
# ============================================================

print("\nSex distribution:")

sex_counts = (
    val_df["SEX"]
    .value_counts()
    .sort_index()
)

print(sex_counts)

valid_sex = (
    val_df["SEX"]
    .isin([1, 2])
)

if not valid_sex.all():
    raise ValueError(
        "Unexpected SEX values detected."
    )


# ============================================================
# STANDARD FEDERATED CHAIN INFERENCE
# ============================================================

X_current = PREPROCESSOR.transform(
    val_df[FEATURES]
)

probabilities = {}
predictions = {}


for target in ORDER:

    model = MODELS[target]

    probability = (
        model.predict_proba(
            X_current
        )[:, 1]
    )

    threshold = (
        THRESHOLDS[target]
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

    # Pass predicted previous label
    # into next classifier in chain.

    X_current = np.column_stack(
        [
            X_current,
            prediction,
        ]
    )


# ============================================================
# GROUP METRIC FUNCTION
# ============================================================

def calculate_group_metrics(
    y_true,
    y_pred,
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    positive_prediction_rate = (
        y_pred.mean()
    )

    prevalence = (
        y_true.mean()
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

    if (fp + tn) > 0:
        false_positive_rate = (
            fp / (fp + tn)
        )
    else:
        false_positive_rate = np.nan

    if (tn + fp) > 0:
        specificity = (
            tn / (tn + fp)
        )
    else:
        specificity = np.nan

    return {
        "n":
            len(y_true),

        "actual_prevalence":
            prevalence,

        "predicted_positive_rate":
            positive_prediction_rate,

        "recall_tpr":
            recall,

        "false_positive_rate":
            false_positive_rate,

        "specificity":
            specificity,

        "precision":
            precision,

        "f1":
            f1,
    }


# ============================================================
# FAIRNESS AUDIT
# ============================================================

GROUPS = {
    "Male": 1,
    "Female": 2,
}

group_rows = []
gap_rows = []


for target in TARGETS:

    print("\n" + "=" * 88)
    print(f"FAIRNESS AUDIT: {target}")
    print("=" * 88)

    target_group_metrics = {}

    for group_name, sex_code in (
        GROUPS.items()
    ):

        mask = (
            val_df["SEX"]
            == sex_code
        )

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
                mask.to_numpy()
            ]
        )

        metrics = (
            calculate_group_metrics(
                y_true,
                y_pred,
            )
        )

        target_group_metrics[
            group_name
        ] = metrics

        row = {
            "target":
                target,

            "group":
                group_name,

            **metrics,
        }

        group_rows.append(
            row
        )

        print(
            f"\n{group_name}"
        )

        print(
            f"Patients                 : "
            f"{metrics['n']}"
        )

        print(
            f"Actual prevalence        : "
            f"{metrics['actual_prevalence'] * 100:.2f}%"
        )

        print(
            f"Predicted positive rate  : "
            f"{metrics['predicted_positive_rate'] * 100:.2f}%"
        )

        print(
            f"Recall / TPR             : "
            f"{metrics['recall_tpr']:.4f}"
        )

        print(
            f"False Positive Rate      : "
            f"{metrics['false_positive_rate']:.4f}"
        )

        print(
            f"Precision                : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"F1                       : "
            f"{metrics['f1']:.4f}"
        )


    male = (
        target_group_metrics[
            "Male"
        ]
    )

    female = (
        target_group_metrics[
            "Female"
        ]
    )


    # --------------------------------------------------------
    # FAIRNESS DIFFERENCES
    # --------------------------------------------------------

    demographic_parity_difference = (
        female[
            "predicted_positive_rate"
        ]
        -
        male[
            "predicted_positive_rate"
        ]
    )

    equal_opportunity_difference = (
        female[
            "recall_tpr"
        ]
        -
        male[
            "recall_tpr"
        ]
    )

    fpr_difference = (
        female[
            "false_positive_rate"
        ]
        -
        male[
            "false_positive_rate"
        ]
    )


    gap_row = {
        "target":
            target,

        "demographic_parity_difference":
            demographic_parity_difference,

        "abs_demographic_parity_difference":
            abs(
                demographic_parity_difference
            ),

        "equal_opportunity_difference":
            equal_opportunity_difference,

        "abs_equal_opportunity_difference":
            abs(
                equal_opportunity_difference
            ),

        "false_positive_rate_difference":
            fpr_difference,

        "abs_false_positive_rate_difference":
            abs(
                fpr_difference
            ),
    }

    gap_rows.append(
        gap_row
    )


    print("\nFairness gaps:")
    print(
        "Female - Male"
    )

    print(
        f"Demographic Parity Difference : "
        f"{demographic_parity_difference:+.4f}"
    )

    print(
        f"Equal Opportunity Difference  : "
        f"{equal_opportunity_difference:+.4f}"
    )

    print(
        f"False Positive Rate Difference: "
        f"{fpr_difference:+.4f}"
    )


# ============================================================
# CREATE DATAFRAMES
# ============================================================

group_df = pd.DataFrame(
    group_rows
)

gap_df = pd.DataFrame(
    gap_rows
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 88)
print("FAIRNESS GAP SUMMARY")
print("=" * 88)

summary_columns = [
    "target",
    "demographic_parity_difference",
    "abs_demographic_parity_difference",
    "equal_opportunity_difference",
    "abs_equal_opportunity_difference",
    "false_positive_rate_difference",
]

print(
    gap_df[
        summary_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# MEAN ABSOLUTE FAIRNESS GAPS
# ============================================================

mean_abs_dp = (
    gap_df[
        "abs_demographic_parity_difference"
    ].mean()
)

mean_abs_eo = (
    gap_df[
        "abs_equal_opportunity_difference"
    ].mean()
)

mean_abs_fpr = (
    gap_df[
        "abs_false_positive_rate_difference"
    ].mean()
)


print("\nMean absolute gaps across 3 diseases:")

print(
    f"Demographic Parity : "
    f"{mean_abs_dp:.4f}"
)

print(
    f"Equal Opportunity  : "
    f"{mean_abs_eo:.4f}"
)

print(
    f"False Positive Rate: "
    f"{mean_abs_fpr:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

group_path = (
    RESULTS_DIR /
    "fairness_baseline_group_metrics.csv"
)

gap_path = (
    RESULTS_DIR /
    "fairness_baseline_gaps.csv"
)

group_df.to_csv(
    group_path,
    index=False,
)

gap_df.to_csv(
    gap_path,
    index=False,
)


print("\nSaved:")
print(group_path)
print(gap_path)


print("\nIMPORTANT:")
print(
    "SEX was used only to AUDIT model behaviour."
)

print(
    "SEX was not supplied to the disease "
    "prediction models."
)

print(
    "These results describe group differences; "
    "they do not by themselves establish whether "
    "the model is legally or clinically fair."
)

print(
    "TEST dataset was NOT loaded or used."
)


print("\n" + "=" * 88)
print("STAGE 5.1 BASELINE FAIRNESS AUDIT COMPLETE.")
print("=" * 88)