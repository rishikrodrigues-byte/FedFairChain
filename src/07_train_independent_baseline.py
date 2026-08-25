from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

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

TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


FEATURES = [
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
]


TARGETS = [
    "DIABETES",
    "HYPERTENSION",
    "CVD",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("FEDFAIRCHAIN - INDEPENDENT CENTRALIZED BASELINE")
print("=" * 78)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print(f"\nTraining rows   : {len(train_df)}")
print(f"Validation rows : {len(val_df)}")
print(f"Predictors      : {len(FEATURES)}")


# ============================================================
# LEAKAGE CHECK
# ============================================================

target_overlap = set(FEATURES).intersection(TARGETS)

if target_overlap:
    raise ValueError(
        f"Target leakage detected: {target_overlap}"
    )

if "SEX" in FEATURES:
    raise ValueError(
        "SEX must not be included in the predictor list."
    )

print("\nLeakage checks:")
print("[PASS] No target columns used as predictors")
print("[PASS] SEX excluded from predictors")


# ============================================================
# INPUT MATRICES
# ============================================================

X_train = train_df[FEATURES]
X_val = val_df[FEATURES]


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(y_true, probabilities):

    thresholds = np.arange(
        0.05,
        0.951,
        0.01
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold)


# ============================================================
# TRAIN ONE MODEL PER TARGET
# ============================================================

all_results = []
thresholds = {}


for target in TARGETS:

    print("\n" + "=" * 78)
    print(f"TRAINING TARGET: {target}")
    print("=" * 78)

    y_train = train_df[target].astype(int)
    y_val = val_df[target].astype(int)

    print(
        f"Training positives: "
        f"{int(y_train.sum())} / {len(y_train)} "
        f"({y_train.mean() * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=0.0001,
                    max_iter=3000,
                    tol=1e-4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION PROBABILITIES
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------------
    # FIND VALIDATION THRESHOLD
    # --------------------------------------------------------

    threshold = find_best_threshold(
        y_val,
        probabilities
    )

    thresholds[target] = threshold

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        predictions
    )

    result = {
        "target": target,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }

    all_results.append(result)

    print(f"\nBest validation threshold : {threshold:.2f}")
    print(f"ROC-AUC                   : {roc_auc:.4f}")
    print(f"PR-AUC                    : {pr_auc:.4f}")
    print(f"Precision                 : {precision:.4f}")
    print(f"Recall                    : {recall:.4f}")
    print(f"F1-score                  : {f1:.4f}")
    print(
        f"Balanced Accuracy         : "
        f"{balanced_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR /
        f"independent_{target.lower()}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"\nSaved model: {model_path}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_path = (
    RESULTS_DIR /
    "independent_baseline_validation.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


threshold_path = (
    MODEL_DIR /
    "independent_thresholds.json"
)

with open(
    threshold_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        thresholds,
        file,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("INDEPENDENT BASELINE VALIDATION RESULTS")
print("=" * 78)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:")
print(results_path)
print(threshold_path)

print("\nIMPORTANT:")
print(
    "The TEST set was NOT used during training "
    "or threshold selection."
)

print("\n" + "=" * 78)
print("STAGE 2.2 INDEPENDENT BASELINE COMPLETE.")
print("=" * 78)