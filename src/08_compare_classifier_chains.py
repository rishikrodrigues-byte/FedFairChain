from pathlib import Path
from itertools import permutations
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
    hamming_loss,
    accuracy_score,
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

print("=" * 80)
print("FEDFAIRCHAIN - CLASSIFIER CHAIN ORDER COMPARISON")
print("=" * 80)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print(f"\nTraining rows   : {len(train_df)}")
print(f"Validation rows : {len(val_df)}")
print(f"Predictors      : {len(FEATURES)}")


# ============================================================
# SAFETY / LEAKAGE CHECKS
# ============================================================

target_overlap = set(FEATURES).intersection(TARGETS)

if target_overlap:
    raise ValueError(
        f"Target leakage detected: {target_overlap}"
    )

if "SEX" in FEATURES:
    raise ValueError(
        "SEX must not be used as a predictor."
    )

print("\nLeakage checks:")
print("[PASS] No targets used as standard predictors")
print("[PASS] SEX excluded from predictors")
print("[PASS] TEST set is not loaded")


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = Pipeline(
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
    ]
)

X_train_base = preprocessor.fit_transform(
    train_df[FEATURES]
)

X_val_base = preprocessor.transform(
    val_df[FEATURES]
)

print("\nPreprocessing fitted using TRAINING data only.")
print(f"Processed feature count: {X_train_base.shape[1]}")


# ============================================================
# HELPER
# ============================================================

def short_name(order):

    mapping = {
        "DIABETES": "D",
        "HYPERTENSION": "H",
        "CVD": "C",
    }

    return " -> ".join(
        mapping[target]
        for target in order
    )


# ============================================================
# TRAIN CUSTOM CLASSIFIER CHAIN
# ============================================================

def train_chain(order):

    models = {}

    X_current = X_train_base.copy()

    for position, target in enumerate(order):

        y_train = (
            train_df[target]
            .astype(int)
            .to_numpy()
        )

        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=3000,
            tol=1e-4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        model.fit(
            X_current,
            y_train
        )

        models[target] = model

        # During classifier-chain TRAINING,
        # ground-truth previous labels are used.
        #
        # They are appended only AFTER the current
        # target has been trained.

        X_current = np.column_stack(
            [
                X_current,
                y_train,
            ]
        )

    return models


# ============================================================
# VALIDATION INFERENCE
# ============================================================

def evaluate_chain(
    order,
    models,
    thresholds=None,
):

    probabilities = {}
    predictions = {}

    X_current = X_val_base.copy()

    for target in order:

        model = models[target]

        prob = model.predict_proba(
            X_current
        )[:, 1]

        if thresholds is None:
            threshold = 0.50
        else:
            threshold = thresholds[target]

        pred = (
            prob >= threshold
        ).astype(int)

        probabilities[target] = prob
        predictions[target] = pred

        # During INFERENCE, the true previous label
        # is NOT available.
        #
        # Therefore the previous prediction is
        # passed to the next classifier.

        X_current = np.column_stack(
            [
                X_current,
                pred,
            ]
        )

    return probabilities, predictions


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    probabilities,
    predictions,
):

    per_target = []

    for target in TARGETS:

        y_true = (
            val_df[target]
            .astype(int)
            .to_numpy()
        )

        prob = probabilities[target]
        pred = predictions[target]

        per_target.append(
            {
                "target": target,

                "roc_auc":
                    roc_auc_score(
                        y_true,
                        prob
                    ),

                "pr_auc":
                    average_precision_score(
                        y_true,
                        prob
                    ),

                "precision":
                    precision_score(
                        y_true,
                        pred,
                        zero_division=0
                    ),

                "recall":
                    recall_score(
                        y_true,
                        pred,
                        zero_division=0
                    ),

                "f1":
                    f1_score(
                        y_true,
                        pred,
                        zero_division=0
                    ),

                "balanced_accuracy":
                    balanced_accuracy_score(
                        y_true,
                        pred
                    ),
            }
        )

    metric_df = pd.DataFrame(
        per_target
    )

    y_true_multi = (
        val_df[TARGETS]
        .astype(int)
        .to_numpy()
    )

    y_pred_multi = np.column_stack(
        [
            predictions[target]
            for target in TARGETS
        ]
    )

    summary = {
        "macro_roc_auc":
            metric_df["roc_auc"].mean(),

        "macro_pr_auc":
            metric_df["pr_auc"].mean(),

        "macro_f1":
            metric_df["f1"].mean(),

        "macro_recall":
            metric_df["recall"].mean(),

        "macro_balanced_accuracy":
            metric_df[
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
                y_pred_multi
            ),

        "subset_accuracy":
            accuracy_score(
                y_true_multi,
                y_pred_multi
            ),
    }

    return metric_df, summary


# ============================================================
# TEST ALL SIX POSSIBLE ORDERS
# ============================================================

orders = list(
    permutations(TARGETS)
)

chain_runs = []
summary_rows = []


print("\n" + "=" * 80)
print("EVALUATING ALL 6 CLASSIFIER CHAIN ORDERS")
print("=" * 80)


for number, order in enumerate(
    orders,
    start=1
):

    print(
        f"\n[{number}/6] "
        f"{short_name(order)}"
    )

    models = train_chain(
        order
    )

    probabilities, predictions = (
        evaluate_chain(
            order,
            models,
            thresholds=None,
        )
    )

    target_metrics, summary = (
        calculate_metrics(
            probabilities,
            predictions
        )
    )

    run = {
        "order": order,
        "models": models,
        "target_metrics":
            target_metrics,
        "summary":
            summary,
    }

    chain_runs.append(
        run
    )

    summary_rows.append(
        {
            "order":
                short_name(order),

            "macro_pr_auc":
                summary[
                    "macro_pr_auc"
                ],

            "macro_roc_auc":
                summary[
                    "macro_roc_auc"
                ],

            "macro_f1":
                summary[
                    "macro_f1"
                ],

            "macro_recall":
                summary[
                    "macro_recall"
                ],

            "macro_balanced_accuracy":
                summary[
                    "macro_balanced_accuracy"
                ],

            "micro_f1":
                summary[
                    "micro_f1"
                ],

            "hamming_loss":
                summary[
                    "hamming_loss"
                ],

            "subset_accuracy":
                summary[
                    "subset_accuracy"
                ],
        }
    )

    print(
        f"Macro PR-AUC : "
        f"{summary['macro_pr_auc']:.4f}"
    )

    print(
        f"Macro ROC-AUC: "
        f"{summary['macro_roc_auc']:.4f}"
    )

    print(
        f"Macro F1     : "
        f"{summary['macro_f1']:.4f}"
    )


# ============================================================
# ORDER COMPARISON TABLE
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df = summary_df.sort_values(
    by=[
        "macro_pr_auc",
        "macro_f1",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(drop=True)


print("\n" + "=" * 80)
print("CLASSIFIER CHAIN ORDER RANKING")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SELECT BEST ORDER
# ============================================================

best_order_name = (
    summary_df.iloc[0]["order"]
)

best_run = None

for run in chain_runs:

    if short_name(
        run["order"]
    ) == best_order_name:

        best_run = run
        break


best_order = best_run["order"]
best_models = best_run["models"]


print("\n" + "=" * 80)
print("BEST CLASSIFIER CHAIN")
print("=" * 80)

print(
    "Selected order:",
    short_name(best_order)
)

print(
    "Selection criterion: "
    "Highest validation Macro PR-AUC"
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
):

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
# TUNE THRESHOLDS FOR WINNING CHAIN
# ============================================================

best_thresholds = {}

tuned_probabilities = {}
tuned_predictions = {}

X_current = X_val_base.copy()


print("\n" + "=" * 80)
print("TUNING WINNING CHAIN THRESHOLDS")
print("=" * 80)


for target in best_order:

    y_val = (
        val_df[target]
        .astype(int)
        .to_numpy()
    )

    model = best_models[target]

    prob = model.predict_proba(
        X_current
    )[:, 1]

    threshold = find_best_threshold(
        y_val,
        prob
    )

    pred = (
        prob >= threshold
    ).astype(int)

    best_thresholds[target] = (
        threshold
    )

    tuned_probabilities[target] = (
        prob
    )

    tuned_predictions[target] = (
        pred
    )

    print(
        f"{target:<12} "
        f"threshold={threshold:.2f}"
    )

    X_current = np.column_stack(
        [
            X_current,
            pred,
        ]
    )


# ============================================================
# FINAL VALIDATION METRICS FOR BEST CHAIN
# ============================================================

best_target_metrics, best_summary = (
    calculate_metrics(
        tuned_probabilities,
        tuned_predictions
    )
)


print("\n" + "=" * 80)
print("BEST CHAIN - TUNED VALIDATION RESULTS")
print("=" * 80)

print(
    best_target_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\nMulti-label summary:")

print(
    f"Macro ROC-AUC            : "
    f"{best_summary['macro_roc_auc']:.4f}"
)

print(
    f"Macro PR-AUC             : "
    f"{best_summary['macro_pr_auc']:.4f}"
)

print(
    f"Macro F1                 : "
    f"{best_summary['macro_f1']:.4f}"
)

print(
    f"Micro F1                 : "
    f"{best_summary['micro_f1']:.4f}"
)

print(
    f"Macro Recall             : "
    f"{best_summary['macro_recall']:.4f}"
)

print(
    f"Macro Balanced Accuracy  : "
    f"{best_summary['macro_balanced_accuracy']:.4f}"
)

print(
    f"Hamming Loss             : "
    f"{best_summary['hamming_loss']:.4f}"
)

print(
    f"Exact Match Accuracy     : "
    f"{best_summary['subset_accuracy']:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

summary_path = (
    RESULTS_DIR /
    "classifier_chain_order_comparison.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


best_metrics_path = (
    RESULTS_DIR /
    "best_classifier_chain_validation.csv"
)

best_target_metrics.to_csv(
    best_metrics_path,
    index=False
)


order_path = (
    MODEL_DIR /
    "best_chain_order.json"
)

with open(
    order_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "order":
                list(best_order),

            "short_order":
                short_name(
                    best_order
                ),

            "thresholds":
                best_thresholds,
        },
        file,
        indent=4,
    )


# ============================================================
# SAVE COMPLETE BEST CHAIN
# ============================================================

bundle = {
    "features":
        FEATURES,

    "targets":
        TARGETS,

    "order":
        best_order,

    "thresholds":
        best_thresholds,

    "preprocessor":
        preprocessor,

    "models":
        best_models,
}


model_path = (
    MODEL_DIR /
    "best_classifier_chain.joblib"
)

joblib.dump(
    bundle,
    model_path
)


print("\nSaved:")
print(summary_path)
print(best_metrics_path)
print(order_path)
print(model_path)


print("\nIMPORTANT:")
print(
    "The TEST dataset was not loaded "
    "or used anywhere in this experiment."
)


print("\n" + "=" * 80)
print("STAGE 3.1 CLASSIFIER CHAIN COMPARISON COMPLETE.")
print("=" * 80)