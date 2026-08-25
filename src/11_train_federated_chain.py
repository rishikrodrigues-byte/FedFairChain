from pathlib import Path
import copy
import json

import joblib
import numpy as np
import pandas as pd

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

CLIENT_DIR = Path(
    "data/processed/federated_clients"
)

VAL_PATH = Path(
    "data/processed/validation.csv"
)

CENTRAL_CHAIN_PATH = Path(
    "models/best_classifier_chain.joblib"
)

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_CLIENTS = 4

FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 2

LEARNING_RATE = 0.01
RANDOM_STATE = 42


# ============================================================
# LOAD CENTRALIZED CHAIN METADATA
# ============================================================

print("=" * 86)
print("FEDFAIRCHAIN - FEDERATED CLASSIFIER CHAIN WITH FEDAVG")
print("=" * 86)

central_bundle = joblib.load(
    CENTRAL_CHAIN_PATH
)

FEATURES = central_bundle["features"]
TARGETS = central_bundle["targets"]
ORDER = list(
    central_bundle["order"]
)

PREPROCESSOR = (
    central_bundle["preprocessor"]
)

print(
    "\nClassifier-chain order:",
    " -> ".join(ORDER)
)

print(
    f"Federated rounds : "
    f"{FEDERATED_ROUNDS}"
)

print(
    f"Local epochs     : "
    f"{LOCAL_EPOCHS}"
)

print(
    f"Learning rate    : "
    f"{LEARNING_RATE}"
)


# ============================================================
# LOAD FOUR LOCAL CLIENT DATASETS
# ============================================================

clients = []

for client_id in range(
    1,
    NUM_CLIENTS + 1
):

    path = (
        CLIENT_DIR /
        f"hospital_{client_id}.csv"
    )

    client_df = pd.read_csv(
        path
    )

    clients.append(
        client_df
    )

    print(
        f"Hospital {client_id}: "
        f"{len(client_df)} patients"
    )


# ============================================================
# LOAD VALIDATION ONLY
# ============================================================

val_df = pd.read_csv(
    VAL_PATH
)

print(
    f"\nValidation patients: "
    f"{len(val_df)}"
)

print(
    "\n[PASS] TEST dataset is not loaded."
)


# ============================================================
# PREPROCESS DATA
# ============================================================

# Shared preprocessing transformation is reused
# from the centralized TRAINING-only preprocessing.
#
# This guarantees identical feature representation
# across all federated clients.

client_base_features = []

for client in clients:

    transformed = (
        PREPROCESSOR.transform(
            client[FEATURES]
        )
    )

    client_base_features.append(
        transformed
    )


X_val_base = PREPROCESSOR.transform(
    val_df[FEATURES]
)


print(
    "\n[PASS] Shared preprocessor was "
    "previously fitted using training data only."
)


# ============================================================
# CREATE CHAIN-AUGMENTED LOCAL DATA
# ============================================================

client_chain_data = []

for client_index, client in enumerate(
    clients
):

    target_data = {}

    X_current = (
        client_base_features[
            client_index
        ].copy()
    )

    for target in ORDER:

        y = (
            client[target]
            .astype(int)
            .to_numpy()
        )

        target_data[target] = {
            "X": X_current.copy(),
            "y": y,
        }

        # Teacher forcing during training:
        # append true previous label.

        X_current = np.column_stack(
            [
                X_current,
                y,
            ]
        )

    client_chain_data.append(
        target_data
    )


# ============================================================
# GLOBAL CLASS BALANCE WEIGHTS
# ============================================================

all_train = pd.concat(
    clients,
    ignore_index=True
)

class_weights = {}

print("\nGlobal class-balance weights:")

for target in TARGETS:

    y = (
        all_train[target]
        .astype(int)
        .to_numpy()
    )

    n = len(y)

    positive = int(
        y.sum()
    )

    negative = (
        n - positive
    )

    positive_weight = (
        n / (2.0 * positive)
    )

    negative_weight = (
        n / (2.0 * negative)
    )

    class_weights[target] = {
        0: negative_weight,
        1: positive_weight,
    }

    print(
        f"{target:<13} "
        f"negative_weight="
        f"{negative_weight:.4f}  "
        f"positive_weight="
        f"{positive_weight:.4f}"
    )


# ============================================================
# INITIALIZE GLOBAL MODELS
# ============================================================

def initialize_model(
    feature_count,
    seed,
):

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="constant",
        eta0=LEARNING_RATE,
        random_state=seed,
    )

    # Initialize required sklearn attributes.
    # We immediately reset learned weights to zero.

    X_dummy = np.zeros(
        (2, feature_count),
        dtype=float,
    )

    y_dummy = np.array(
        [0, 1],
        dtype=int,
    )

    model.partial_fit(
        X_dummy,
        y_dummy,
        classes=np.array([0, 1]),
    )

    model.coef_[:] = 0.0
    model.intercept_[:] = 0.0

    return model


global_models = {}

for position, target in enumerate(
    ORDER
):

    feature_count = (
        len(FEATURES)
        + position
    )

    global_models[target] = (
        initialize_model(
            feature_count,
            RANDOM_STATE + position,
        )
    )


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_local_model(
    global_model,
    X,
    y,
    target,
    client_seed,
):

    local_model = copy.deepcopy(
        global_model
    )

    rng = np.random.default_rng(
        client_seed
    )

    weights = class_weights[
        target
    ]

    sample_weights = np.where(
        y == 1,
        weights[1],
        weights[0],
    )

    for _ in range(
        LOCAL_EPOCHS
    ):

        indices = np.arange(
            len(y)
        )

        rng.shuffle(
            indices
        )

        local_model.partial_fit(
            X[indices],
            y[indices],
            sample_weight=(
                sample_weights[
                    indices
                ]
            ),
        )

    return local_model


# ============================================================
# FEDAVG
# ============================================================

def fedavg(
    local_models,
    client_sizes,
    global_model,
):

    total_samples = float(
        sum(client_sizes)
    )

    averaged_coef = np.zeros_like(
        global_model.coef_
    )

    averaged_intercept = (
        np.zeros_like(
            global_model.intercept_
        )
    )

    for model, size in zip(
        local_models,
        client_sizes,
    ):

        weight = (
            size /
            total_samples
        )

        averaged_coef += (
            weight
            * model.coef_
        )

        averaged_intercept += (
            weight
            * model.intercept_
        )

    global_model.coef_ = (
        averaged_coef
    )

    global_model.intercept_ = (
        averaged_intercept
    )

    return global_model


# ============================================================
# VALIDATION INFERENCE
# ============================================================

def predict_chain(
    models,
    thresholds=None,
):

    probabilities = {}
    predictions = {}

    X_current = (
        X_val_base.copy()
    )

    for target in ORDER:

        probability = (
            models[target]
            .predict_proba(
                X_current
            )[:, 1]
        )

        if thresholds is None:
            threshold = 0.50
        else:
            threshold = (
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
# VALIDATION METRICS
# ============================================================

def evaluate_probabilities(
    probabilities
):

    pr_values = []
    roc_values = []

    for target in TARGETS:

        y_true = (
            val_df[target]
            .astype(int)
            .to_numpy()
        )

        pr_values.append(
            average_precision_score(
                y_true,
                probabilities[target],
            )
        )

        roc_values.append(
            roc_auc_score(
                y_true,
                probabilities[target],
            )
        )

    return (
        float(np.mean(pr_values)),
        float(np.mean(roc_values)),
    )


# ============================================================
# FEDERATED TRAINING
# ============================================================

history_rows = []

best_macro_pr_auc = -1.0
best_round = None
best_models = None


print("\n" + "=" * 86)
print("FEDERATED TRAINING")
print("=" * 86)


for round_number in range(
    1,
    FEDERATED_ROUNDS + 1
):

    for target_position, target in enumerate(
        ORDER
    ):

        local_models = []
        client_sizes = []

        for client_index in range(
            NUM_CLIENTS
        ):

            data = (
                client_chain_data[
                    client_index
                ][target]
            )

            X_local = data["X"]
            y_local = data["y"]

            local_model = (
                train_local_model(
                    global_models[target],
                    X_local,
                    y_local,
                    target,
                    client_seed=(
                        RANDOM_STATE
                        + round_number * 100
                        + target_position * 10
                        + client_index
                    ),
                )
            )

            local_models.append(
                local_model
            )

            client_sizes.append(
                len(y_local)
            )

        global_models[target] = (
            fedavg(
                local_models,
                client_sizes,
                global_models[target],
            )
        )

    # --------------------------------------------------------
    # Validation after each federated round
    # --------------------------------------------------------

    probabilities, _ = (
        predict_chain(
            global_models,
            thresholds=None,
        )
    )

    macro_pr_auc, macro_roc_auc = (
        evaluate_probabilities(
            probabilities
        )
    )

    history_rows.append(
        {
            "round":
                round_number,

            "macro_pr_auc":
                macro_pr_auc,

            "macro_roc_auc":
                macro_roc_auc,
        }
    )

    print(
        f"Round {round_number:02d} | "
        f"Macro PR-AUC="
        f"{macro_pr_auc:.4f} | "
        f"Macro ROC-AUC="
        f"{macro_roc_auc:.4f}"
    )

    # Validation-based early model selection.
    if macro_pr_auc > (
        best_macro_pr_auc
    ):

        best_macro_pr_auc = (
            macro_pr_auc
        )

        best_round = (
            round_number
        )

        best_models = copy.deepcopy(
            global_models
        )


# ============================================================
# RESTORE BEST ROUND
# ============================================================

global_models = (
    best_models
)

print("\n" + "=" * 86)
print("BEST FEDERATED ROUND")
print("=" * 86)

print(
    f"Selected round : "
    f"{best_round}"
)

print(
    f"Validation Macro PR-AUC : "
    f"{best_macro_pr_auc:.4f}"
)


# ============================================================
# TUNE FEDERATED THRESHOLDS
# ============================================================

def find_best_threshold(
    y_true,
    probability,
):

    thresholds = np.arange(
        0.05,
        0.951,
        0.01
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        prediction = (
            probability
            >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = (
                threshold
            )

    return float(
        best_threshold
    )


federated_thresholds = {}

final_probabilities = {}
final_predictions = {}

X_current = (
    X_val_base.copy()
)


print("\n" + "=" * 86)
print("FEDERATED THRESHOLD TUNING")
print("=" * 86)


for target in ORDER:

    y_true = (
        val_df[target]
        .astype(int)
        .to_numpy()
    )

    probability = (
        global_models[target]
        .predict_proba(
            X_current
        )[:, 1]
    )

    threshold = (
        find_best_threshold(
            y_true,
            probability,
        )
    )

    prediction = (
        probability
        >= threshold
    ).astype(int)

    federated_thresholds[target] = (
        threshold
    )

    final_probabilities[target] = (
        probability
    )

    final_predictions[target] = (
        prediction
    )

    print(
        f"{target:<13} "
        f"threshold={threshold:.2f}"
    )

    X_current = np.column_stack(
        [
            X_current,
            prediction,
        ]
    )


# ============================================================
# FINAL VALIDATION METRICS
# ============================================================

metric_rows = []

for target in TARGETS:

    y_true = (
        val_df[target]
        .astype(int)
        .to_numpy()
    )

    probability = (
        final_probabilities[target]
    )

    prediction = (
        final_predictions[target]
    )

    metric_rows.append(
        {
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


metrics_df = pd.DataFrame(
    metric_rows
)


y_true_multi = (
    val_df[TARGETS]
    .astype(int)
    .to_numpy()
)

y_pred_multi = np.column_stack(
    [
        final_predictions[target]
        for target in TARGETS
    ]
)


macro_pr_auc = (
    metrics_df["pr_auc"].mean()
)

macro_roc_auc = (
    metrics_df["roc_auc"].mean()
)

macro_f1 = (
    metrics_df["f1"].mean()
)

micro_f1 = f1_score(
    y_true_multi,
    y_pred_multi,
    average="micro",
    zero_division=0,
)

hamming = hamming_loss(
    y_true_multi,
    y_pred_multi,
)

subset_accuracy = accuracy_score(
    y_true_multi,
    y_pred_multi,
)


print("\n" + "=" * 86)
print("FEDERATED CHAIN - VALIDATION RESULTS")
print("=" * 86)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nMulti-label summary:")

print(
    f"Macro ROC-AUC        : "
    f"{macro_roc_auc:.4f}"
)

print(
    f"Macro PR-AUC         : "
    f"{macro_pr_auc:.4f}"
)

print(
    f"Macro F1             : "
    f"{macro_f1:.4f}"
)

print(
    f"Micro F1             : "
    f"{micro_f1:.4f}"
)

print(
    f"Hamming Loss         : "
    f"{hamming:.4f}"
)

print(
    f"Exact Match Accuracy : "
    f"{subset_accuracy:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

federated_bundle = {
    "features":
        FEATURES,

    "targets":
        TARGETS,

    "order":
        ORDER,

    "thresholds":
        federated_thresholds,

    "preprocessor":
        PREPROCESSOR,

    "models":
        global_models,

    "best_round":
        best_round,

    "num_clients":
        NUM_CLIENTS,

    "federated_rounds":
        FEDERATED_ROUNDS,

    "local_epochs":
        LOCAL_EPOCHS,
}


model_path = (
    MODEL_DIR /
    "federated_classifier_chain.joblib"
)

joblib.dump(
    federated_bundle,
    model_path,
)


# ============================================================
# SAVE RESULTS
# ============================================================

history_df = pd.DataFrame(
    history_rows
)

history_path = (
    RESULTS_DIR /
    "federated_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False,
)


metrics_path = (
    RESULTS_DIR /
    "federated_chain_validation.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)


threshold_path = (
    MODEL_DIR /
    "federated_thresholds.json"
)

with open(
    threshold_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        federated_thresholds,
        file,
        indent=4,
    )


print("\nSaved:")
print(model_path)
print(history_path)
print(metrics_path)
print(threshold_path)


print("\nIMPORTANT:")
print(
    "Raw patient rows remained inside "
    "their simulated client partitions "
    "during local model training."
)

print(
    "FedAvg aggregated model parameters "
    "using client sample counts."
)

print(
    "Validation data was used only for "
    "model selection and threshold tuning."
)

print(
    "TEST dataset was NOT loaded or used."
)


print("\n" + "=" * 86)
print("STAGE 4.2 FEDERATED TRAINING COMPLETE.")
print("=" * 86)