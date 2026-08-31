from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ============================================================
# CONFIGURATION
# ============================================================

VAL_PATH = Path(
    "data/processed/validation.csv"
)

MODEL_PATH = Path(
    "models/fair_federated_classifier_chain.joblib"
)

RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42
BACKGROUND_SIZE = 100
TOP_FEATURES = 10


# ============================================================
# LOAD MODEL + VALIDATION DATA
# ============================================================

print("=" * 92)
print("FEDFAIRCHAIN - STAGE 6 SHAP EXPLAINABILITY")
print("=" * 92)

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
    f"\nValidation patients : {len(val_df)}"
)

print(
    "Classifier chain   : "
    + " -> ".join(ORDER)
)

print(
    f"Base predictors     : {len(FEATURES)}"
)

if "SEX" in FEATURES:
    raise ValueError(
        "SEX must not be present in model predictors."
    )

print(
    "\n[PASS] SEX is not used as a model predictor."
)

print(
    "[PASS] TEST dataset is NOT loaded."
)


# ============================================================
# PREPROCESS VALIDATION DATA
# ============================================================

X_base = PREPROCESSOR.transform(
    val_df[FEATURES]
)


# ============================================================
# RUN CHAIN INFERENCE AND STORE EACH MODEL INPUT
# ============================================================

stage_inputs = {}
stage_feature_names = {}

probabilities = {}
predictions = {}

X_current = X_base.copy()

current_feature_names = list(
    FEATURES
)


for target in ORDER:

    # Save exact input seen by this classifier.
    stage_inputs[target] = (
        X_current.copy()
    )

    stage_feature_names[target] = (
        current_feature_names.copy()
    )

    model = MODELS[target]

    probability = (
        model.predict_proba(
            X_current
        )[:, 1]
    )

    prediction = (
        probability
        >= THRESHOLDS[target]
    ).astype(int)

    probabilities[target] = (
        probability
    )

    predictions[target] = (
        prediction
    )


    # --------------------------------------------------------
    # Add the current prediction as context
    # for the next classifier.
    # --------------------------------------------------------

    context_name = (
        f"{target}_CONTEXT"
    )

    X_current = np.column_stack(
        [
            X_current,
            prediction,
        ]
    )

    current_feature_names.append(
        context_name
    )


# ============================================================
# CHECK INPUT DIMENSIONS
# ============================================================

print("\nModel input dimensions:")

for target in ORDER:

    X_stage = (
        stage_inputs[target]
    )

    names = (
        stage_feature_names[target]
    )

    print(
        f"{target:<13}: "
        f"{X_stage.shape[1]} features"
    )

    if X_stage.shape[1] != len(names):
        raise RuntimeError(
            f"Feature-name mismatch for {target}"
        )


# ============================================================
# RANDOM BACKGROUND REFERENCE
#
# We use a fixed subset of validation/reference samples
# only for post-hoc SHAP explanation.
#
# No model fitting is performed here.
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

background_count = min(
    BACKGROUND_SIZE,
    len(val_df)
)

background_indices = (
    rng.choice(
        len(val_df),
        size=background_count,
        replace=False,
    )
)


# ============================================================
# SHAP EXPLANATION
# ============================================================

global_rows = []

shap_objects = {}
explanation_metadata = {}


print("\n" + "=" * 92)
print("COMPUTING SHAP VALUES")
print("=" * 92)


for target in ORDER:

    print(
        f"\nExplaining: {target}"
    )

    model = (
        MODELS[target]
    )

    X_stage = (
        stage_inputs[target]
    )

    feature_names = (
        stage_feature_names[target]
    )

    background = (
        X_stage[
            background_indices
        ]
    )


    # --------------------------------------------------------
    # LINEAR SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = (
        shap.LinearExplainer(
            model,
            background,
        )
    )

    shap_values = (
        explainer(
            X_stage
        )
    )


    # --------------------------------------------------------
    # ENSURE 2-D SHAP ARRAY
    # --------------------------------------------------------

    values = np.asarray(
        shap_values.values
    )

    if values.ndim != 2:
        raise RuntimeError(
            f"Unexpected SHAP dimensions "
            f"for {target}: {values.shape}"
        )


    shap_objects[target] = {
        "values":
            values,

        "base_values":
            shap_values.base_values,

        "feature_names":
            feature_names,
    }


    # --------------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # Mean absolute SHAP value.
    # --------------------------------------------------------

    mean_abs_shap = np.mean(
        np.abs(values),
        axis=0,
    )


    importance_df = pd.DataFrame(
        {
            "target":
                target,

            "feature":
                feature_names,

            "mean_abs_shap":
                mean_abs_shap,
        }
    )


    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )


    print(
        "\nTop SHAP features:"
    )

    print(
        importance_df.head(
            TOP_FEATURES
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


    global_rows.append(
        importance_df
    )


    # --------------------------------------------------------
    # SAVE TARGET CSV
    # --------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        /
        f"shap_{target.lower()}_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False,
    )


    # --------------------------------------------------------
    # CREATE GLOBAL IMPORTANCE GRAPH
    # --------------------------------------------------------

    plot_df = (
        importance_df
        .head(TOP_FEATURES)
        .sort_values(
            "mean_abs_shap",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["mean_abs_shap"],
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{target} - Global SHAP Feature Importance"
    )

    plt.tight_layout()


    plot_path = (
        RESULTS_DIR
        /
        f"shap_{target.lower()}_global.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # --------------------------------------------------------
    # ADDITIVITY CHECK
    #
    # For the linear logistic model:
    #
    # expected value + sum(SHAP values)
    # should reproduce the raw model score
    # (log-odds / decision function).
    # --------------------------------------------------------

    raw_scores = (
        model.decision_function(
            X_stage
        )
    )

    base_values = np.asarray(
        shap_values.base_values
    )

    if base_values.ndim == 0:

        reconstructed = (
            float(base_values)
            +
            values.sum(axis=1)
        )

    elif base_values.ndim == 1:

        reconstructed = (
            base_values
            +
            values.sum(axis=1)
        )

    else:

        raise RuntimeError(
            f"Unexpected base-value shape "
            f"for {target}: "
            f"{base_values.shape}"
        )


    max_error = float(
        np.max(
            np.abs(
                raw_scores
                - reconstructed
            )
        )
    )


    print(
        f"\nSHAP additivity max error: "
        f"{max_error:.10f}"
    )

    if max_error < 1e-5:

        print(
            "[PASS] SHAP explanation "
            "reconstructs model output."
        )

    else:

        print(
            "[WARNING] SHAP additivity "
            "error is larger than expected."
        )


    explanation_metadata[target] = {
        "feature_names":
            feature_names,

        "max_additivity_error":
            max_error,

        "importance_csv":
            str(csv_path),

        "importance_plot":
            str(plot_path),
    }


# ============================================================
# COMBINED GLOBAL IMPORTANCE
# ============================================================

combined_importance = pd.concat(
    global_rows,
    ignore_index=True,
)

combined_path = (
    RESULTS_DIR
    /
    "shap_global_importance_all_targets.csv"
)

combined_importance.to_csv(
    combined_path,
    index=False,
)


# ============================================================
# SELECT ONE EXAMPLE PATIENT
#
# Select validation participant with highest
# average predicted disease probability.
# ============================================================

combined_probability = np.mean(
    np.column_stack(
        [
            probabilities[target]
            for target in TARGETS
        ]
    ),
    axis=1,
)

example_index = int(
    np.argmax(
        combined_probability
    )
)

example_seqn = (
    val_df.iloc[
        example_index
    ]["SEQN"]
)


print("\n" + "=" * 92)
print("EXAMPLE LOCAL PATIENT EXPLANATION")
print("=" * 92)

print(
    f"\nValidation row index : "
    f"{example_index}"
)

print(
    f"SEQN                 : "
    f"{example_seqn}"
)


local_rows = []


for target in ORDER:

    values = (
        shap_objects[target][
            "values"
        ][example_index]
    )

    names = (
        shap_objects[target][
            "feature_names"
        ]
    )

    probability = float(
        probabilities[target][
            example_index
        ]
    )

    prediction = int(
        predictions[target][
            example_index
        ]
    )

    contributions = pd.DataFrame(
        {
            "feature":
                names,

            "shap_value":
                values,
        }
    )

    contributions[
        "abs_shap_value"
    ] = np.abs(
        contributions[
            "shap_value"
        ]
    )

    contributions = (
        contributions
        .sort_values(
            "abs_shap_value",
            ascending=False,
        )
        .reset_index(drop=True)
    )


    print(
        f"\n{target}"
    )

    print(
        f"Probability : "
        f"{probability:.4f}"
    )

    print(
        f"Prediction  : "
        f"{prediction}"
    )

    print(
        f"Threshold   : "
        f"{THRESHOLDS[target]:.2f}"
    )

    print(
        "\nTop local contributors:"
    )


    top_local = (
        contributions.head(5)
    )

    for _, row in (
        top_local.iterrows()
    ):

        direction = (
            "increases"
            if row["shap_value"] > 0
            else "decreases"
        )

        print(
            f"  {row['feature']:<25} "
            f"{row['shap_value']:+.4f} "
            f"({direction} model score)"
        )


        local_rows.append(
            {
                "SEQN":
                    example_seqn,

                "target":
                    target,

                "probability":
                    probability,

                "prediction":
                    prediction,

                "threshold":
                    THRESHOLDS[target],

                "feature":
                    row["feature"],

                "shap_value":
                    row["shap_value"],

                "direction":
                    direction,
            }
        )


# ============================================================
# SAVE LOCAL EXPLANATION
# ============================================================

local_df = pd.DataFrame(
    local_rows
)

local_path = (
    RESULTS_DIR
    /
    "shap_example_patient.csv"
)

local_df.to_csv(
    local_path,
    index=False,
)


# ============================================================
# SAVE BACKGROUND REFERENCE FOR FUTURE APP
# ============================================================

background_bundle = {}

for target in ORDER:

    background_bundle[target] = (
        stage_inputs[target][
            background_indices
        ]
    )


background_path = (
    MODEL_DIR
    /
    "shap_backgrounds.joblib"
)

joblib.dump(
    {
        "backgrounds":
            background_bundle,

        "feature_names":
            stage_feature_names,

        "background_size":
            background_count,

        "random_state":
            RANDOM_STATE,
    },
    background_path,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = (
    RESULTS_DIR
    /
    "shap_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "model":
                str(MODEL_PATH),

            "chain_order":
                ORDER,

            "background_size":
                background_count,

            "example_seqn":
                float(example_seqn),

            "targets":
                explanation_metadata,
        },
        file,
        indent=4,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 92)
print("SHAP OUTPUT FILES")
print("=" * 92)

print(
    "\nCombined global importance:"
)

print(
    combined_path
)

print(
    "\nExample patient explanation:"
)

print(
    local_path
)

print(
    "\nSaved SHAP background reference:"
)

print(
    background_path
)

print(
    "\nGlobal plots:"
)

for target in ORDER:

    print(
        RESULTS_DIR
        /
        f"shap_{target.lower()}_global.png"
    )


print("\nIMPORTANT:")

print(
    "SHAP values explain the model's "
    "raw decision score (log-odds), "
    "not direct percentage-point changes "
    "in disease probability."
)

print(
    "Positive SHAP values push the prediction "
    "toward the positive class."
)

print(
    "Negative SHAP values push the prediction "
    "toward the negative class."
)

print(
    "Context features such as DIABETES_CONTEXT "
    "represent previous classifier-chain outputs."
)

print(
    "TEST dataset was NOT loaded or used."
)


print("\n" + "=" * 92)
print("STAGE 6 SHAP EXPLAINABILITY COMPLETE.")
print("=" * 92)