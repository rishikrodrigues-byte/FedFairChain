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

BASE_MODEL_PATH = Path(
    "models/federated_classifier_chain.joblib"
)

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 4

FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 2

LEARNING_RATE = 0.01
RANDOM_STATE = 42


# ============================================================
# LOAD FEDERATED MODEL METADATA
# ============================================================

print("=" * 90)
print("FEDFAIRCHAIN - FAIRNESS-AWARE FEDERATED TRAINING")
print("=" * 90)

base_bundle = joblib.load(
    BASE_MODEL_PATH
)

FEATURES = base_bundle["features"]
TARGETS = base_bundle["targets"]
ORDER = list(base_bundle["order"])
PREPROCESSOR = base_bundle["preprocessor"]

print(
    "\nClassifier chain:",
    " -> ".join(ORDER)
)

print(
    f"Federated rounds : {FEDERATED_ROUNDS}"
)

print(
    f"Local epochs     : {LOCAL_EPOCHS}"
)

print(
    f"Learning rate    : {LEARNING_RATE}"
)

if "SEX" in FEATURES:
    raise ValueError(
        "SEX must NOT be a prediction feature."
    )

print(
    "\n[PASS] SEX is excluded from predictors."
)


# ============================================================
# LOAD LOCAL HOSPITAL DATA
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

    client_df = pd.read_csv(path)

    clients.append(client_df)

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
    "[PASS] TEST dataset is NOT loaded."
)


# ============================================================
# PREPROCESS LOCAL DATA
# ============================================================

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
    "[PASS] Existing training-only "
    "preprocessor reused."
)


# ============================================================
# CREATE CLASSIFIER-CHAIN TRAINING DATA
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

        sex = (
            client["SEX"]
            .astype(int)
            .to_numpy()
        )

        target_data[target] = {
            "X": X_current.copy(),
            "y": y,
            "sex": sex,
        }

        # Teacher forcing:
        # true previous label during training.
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
# GLOBAL AGGREGATED COUNTS
#
# We calculate only aggregated counts across hospitals.
# Raw rows are not pooled into one training dataframe.
# ============================================================

total_samples = sum(
    len(client)
    for client in clients
)

class_weights = {}
fairness_weights = {}


print("\n" + "=" * 90)
print("GLOBAL CLASS + FAIRNESS WEIGHTS")
print("=" * 90)


for target in TARGETS:

    positive = 0
    negative = 0

    sex_counts = {
        1: 0,
        2: 0,
    }

    label_counts = {
        0: 0,
        1: 0,
    }

    joint_counts = {
        (1, 0): 0,
        (1, 1): 0,
        (2, 0): 0,
        (2, 1): 0,
    }


    # --------------------------------------------------------
    # Aggregate counts from each hospital
    # --------------------------------------------------------

    for client in clients:

        y = (
            client[target]
            .astype(int)
        )

        sex = (
            client["SEX"]
            .astype(int)
        )

        positive += int(
            (y == 1).sum()
        )

        negative += int(
            (y == 0).sum()
        )

        for sex_code in [1, 2]:

            sex_counts[sex_code] += int(
                (sex == sex_code).sum()
            )

        for label in [0, 1]:

            label_counts[label] += int(
                (y == label).sum()
            )

        for sex_code in [1, 2]:

            for label in [0, 1]:

                joint_counts[
                    (sex_code, label)
                ] += int(
                    (
                        (sex == sex_code)
                        &
                        (y == label)
                    ).sum()
                )


    # --------------------------------------------------------
    # Class balancing
    # --------------------------------------------------------

    class_weights[target] = {
        0:
            total_samples
            / (2.0 * negative),

        1:
            total_samples
            / (2.0 * positive),
    }


    # --------------------------------------------------------
    # Fairness reweighing
    #
    # w(A=a, Y=y)
    # =
    # P(A=a) * P(Y=y)
    # ----------------
    # P(A=a, Y=y)
    #
    # This reduces dependence between protected
    # group and target outcome in weighted training.
    # --------------------------------------------------------

    target_fair_weights = {}

    for sex_code in [1, 2]:

        for label in [0, 1]:

            joint = joint_counts[
                (sex_code, label)
            ]

            if joint == 0:
                raise ValueError(
                    f"Zero count for "
                    f"{target}, SEX={sex_code}, "
                    f"label={label}"
                )

            weight = (
                sex_counts[sex_code]
                * label_counts[label]
            ) / (
                total_samples
                * joint
            )

            target_fair_weights[
                (sex_code, label)
            ] = float(weight)


    fairness_weights[target] = (
        target_fair_weights
    )


    print(f"\n{target}")

    print(
        "Class weights:"
    )

    print(
        f"  Negative = "
        f"{class_weights[target][0]:.4f}"
    )

    print(
        f"  Positive = "
        f"{class_weights[target][1]:.4f}"
    )

    print(
        "Fairness reweighing:"
    )

    print(
        f"  Male, Negative   = "
        f"{target_fair_weights[(1, 0)]:.4f}"
    )

    print(
        f"  Male, Positive   = "
        f"{target_fair_weights[(1, 1)]:.4f}"
    )

    print(
        f"  Female, Negative = "
        f"{target_fair_weights[(2, 0)]:.4f}"
    )

    print(
        f"  Female, Positive = "
        f"{target_fair_weights[(2, 1)]:.4f}"
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
# FAIRNESS-AWARE LOCAL TRAINING
# ============================================================

def train_local_model(
    global_model,
    X,
    y,
    sex,
    target,
    client_seed,
):

    local_model = copy.deepcopy(
        global_model
    )

    rng = np.random.default_rng(
        client_seed
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_sample_weight = np.where(
        y == 1,
        class_weights[target][1],
        class_weights[target][0],
    )


    # --------------------------------------------------------
    # Fairness weights
    # --------------------------------------------------------

    fairness_sample_weight = np.array(
        [
            fairness_weights[target][
                (int(sex_value), int(label))
            ]
            for sex_value, label
            in zip(sex, y)
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # Combine weights
    # --------------------------------------------------------

    sample_weight = (
        class_sample_weight
        * fairness_sample_weight
    )


    # Normalize to mean 1 for stable SGD.
    sample_weight = (
        sample_weight
        / sample_weight.mean()
    )


    for _ in range(
        LOCAL_EPOCHS
    ):

        indices = np.arange(
            len(y)
        )

        rng.shuffle(indices)

        local_model.partial_fit(
            X[indices],
            y[indices],
            sample_weight=(
                sample_weight[
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

    total = float(
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

        contribution = (
            size / total
        )

        averaged_coef += (
            contribution
            * model.coef_
        )

        averaged_intercept += (
            contribution
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
# PROBABILITY METRICS FOR ROUND SELECTION
# ============================================================

def evaluate_probabilities(
    probabilities
):

    pr_scores = []
    roc_scores = []

    for target in TARGETS:

        y_true = (
            val_df[target]
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

    return (
        float(np.mean(pr_scores)),
        float(np.mean(roc_scores)),
    )


# ============================================================
# FEDERATED TRAINING
# ============================================================

history_rows = []

best_macro_pr_auc = -1.0
best_round = None
best_models = None


print("\n" + "=" * 90)
print("FAIRNESS-AWARE FEDERATED TRAINING")
print("=" * 90)


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

            local_model = (
                train_local_model(
                    global_models[target],
                    data["X"],
                    data["y"],
                    data["sex"],
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
                len(data["y"])
            )


        global_models[target] = (
            fedavg(
                local_models,
                client_sizes,
                global_models[target],
            )
        )


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


    if macro_pr_auc > best_macro_pr_auc:

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

global_models = best_models


print("\n" + "=" * 90)
print("BEST FAIRNESS-AWARE FEDERATED ROUND")
print("=" * 90)

print(
    f"Selected round : "
    f"{best_round}"
)

print(
    f"Validation Macro PR-AUC : "
    f"{best_macro_pr_auc:.4f}"
)


# ============================================================
# THRESHOLD TUNING
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
            best_threshold = threshold

    return float(
        best_threshold
    )


fair_thresholds = {}

final_probabilities = {}
final_predictions = {}

X_current = (
    X_val_base.copy()
)


print("\n" + "=" * 90)
print("FAIRNESS-AWARE THRESHOLD TUNING")
print("=" * 90)


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

    fair_thresholds[target] = (
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
# FINAL VALIDATION PERFORMANCE
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


macro_roc_auc = (
    metrics_df["roc_auc"].mean()
)

macro_pr_auc = (
    metrics_df["pr_auc"].mean()
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


print("\n" + "=" * 90)
print("FAIRNESS-AWARE FEDERATED VALIDATION RESULTS")
print("=" * 90)

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
# SAVE FAIRNESS-AWARE MODEL
# ============================================================

fair_bundle = {
    "features":
        FEATURES,

    "targets":
        TARGETS,

    "order":
        ORDER,

    "thresholds":
        fair_thresholds,

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

    "fairness_method":
        "Sex-label reweighing",

    "protected_attribute":
        "SEX",

    "fairness_weights":
        fairness_weights,
}


model_path = (
    MODEL_DIR /
    "fair_federated_classifier_chain.joblib"
)

joblib.dump(
    fair_bundle,
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
    "fair_federated_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False,
)


metrics_path = (
    RESULTS_DIR /
    "fair_federated_validation.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
)


threshold_path = (
    MODEL_DIR /
    "fair_federated_thresholds.json"
)

with open(
    threshold_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        fair_thresholds,
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
    "SEX was used only to calculate "
    "fairness-aware training weights."
)

print(
    "SEX is NOT an input to the "
    "disease prediction models."
)

print(
    "FedAvg still aggregates local "
    "model parameters using hospital size."
)

print(
    "The TEST dataset was NOT loaded "
    "or used."
)


print("\n" + "=" * 90)
print("STAGE 5.2 FAIRNESS-AWARE FEDERATED TRAINING COMPLETE.")
print("=" * 90)