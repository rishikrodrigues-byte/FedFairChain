from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT / "src"
APP_DIR = ROOT / "app"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "final_figures"


# ============================================================
# COUNTERS
# ============================================================

passed = 0
failed = 0
warnings = 0


def pass_check(message):
    global passed
    passed += 1
    print(f"[PASS] {message}")


def fail_check(message):
    global failed
    failed += 1
    print(f"[FAIL] {message}")


def warn_check(message):
    global warnings
    warnings += 1
    print(f"[WARN] {message}")


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


print("=" * 90)
print("FEDFAIRCHAIN - FINAL PROJECT AUDIT")
print("=" * 90)

print(
    "\nThis script performs verification only."
)

print(
    "No model training, tuning or fitting is performed."
)


# ============================================================
# 1. CORE SOURCE FILES
# ============================================================

section("1. SOURCE CODE CHECK")


required_source_files = [
    "01_download_nhanes.py",
    "02_verify_nhanes.py",
    "03_inspect_nhanes_schema.py",
    "04_build_patient_dataset.py",
    "05_audit_stage1.py",
    "06_create_splits.py",
    "07_train_independent_baseline.py",
    "08_compare_classifier_chains.py",
    "09_diagnosis_aware_chain.py",
    "10_create_federated_clients.py",
    "11_train_federated_chain.py",
    "12_fairness_audit.py",
    "13_train_fair_federated_chain.py",
    "14_compare_fairness.py",
    "15_shap_explainability.py",
    "16_final_test_evaluation.py",
    "17_final_diagnosis_aware_test.py",
    "18_generate_final_figures.py",
]


for filename in required_source_files:

    path = SRC_DIR / filename

    if path.exists():
        pass_check(filename)
    else:
        fail_check(
            f"Missing source file: {filename}"
        )


# ============================================================
# 2. APPLICATION FILES
# ============================================================

section("2. APPLICATION CHECK")


required_app_files = [
    "predictor.py",
    "main.py",
    "index.html",
]


for filename in required_app_files:

    path = APP_DIR / filename

    if path.exists():
        pass_check(
            f"app/{filename}"
        )
    else:
        fail_check(
            f"Missing app/{filename}"
        )


# ============================================================
# 3. DATASET FILES
# ============================================================

section("3. DATASET CHECK")


dataset_files = {
    "Master dataset":
        DATA_DIR /
        "fedfairchain_master.csv",

    "Training split":
        DATA_DIR /
        "train.csv",

    "Validation split":
        DATA_DIR /
        "validation.csv",

    "Final test split":
        DATA_DIR /
        "test.csv",
}


dataframes = {}


for name, path in dataset_files.items():

    if not path.exists():

        fail_check(
            f"{name} missing: {path}"
        )

        continue


    df = pd.read_csv(path)

    dataframes[name] = df

    pass_check(
        f"{name}: "
        f"{len(df)} rows, "
        f"{len(df.columns)} columns"
    )


# ============================================================
# 4. SPLIT INTEGRITY
# ============================================================

section("4. TRAIN / VALIDATION / TEST INTEGRITY")


if all(
    name in dataframes
    for name in [
        "Training split",
        "Validation split",
        "Final test split",
    ]
):

    train_df = dataframes[
        "Training split"
    ]

    val_df = dataframes[
        "Validation split"
    ]

    test_df = dataframes[
        "Final test split"
    ]


    expected_sizes = {
        "Training split": 6056,
        "Validation split": 1298,
        "Final test split": 1298,
    }


    for name, expected in (
        expected_sizes.items()
    ):

        actual = len(
            dataframes[name]
        )

        if actual == expected:

            pass_check(
                f"{name} size = {actual}"
            )

        else:

            warn_check(
                f"{name} size is {actual}; "
                f"expected {expected}"
            )


    train_ids = set(
        train_df["SEQN"]
    )

    val_ids = set(
        val_df["SEQN"]
    )

    test_ids = set(
        test_df["SEQN"]
    )


    if not (
        train_ids & val_ids
    ):

        pass_check(
            "No Train ↔ Validation overlap"
        )

    else:

        fail_check(
            "Train ↔ Validation overlap detected"
        )


    if not (
        train_ids & test_ids
    ):

        pass_check(
            "No Train ↔ Test overlap"
        )

    else:

        fail_check(
            "Train ↔ Test overlap detected"
        )


    if not (
        val_ids & test_ids
    ):

        pass_check(
            "No Validation ↔ Test overlap"
        )

    else:

        fail_check(
            "Validation ↔ Test overlap detected"
        )


    total_unique = len(
        train_ids
        | val_ids
        | test_ids
    )


    if total_unique == 8652:

        pass_check(
            "All 8,652 final participants "
            "are represented exactly once"
        )

    else:

        warn_check(
            f"Unique split participants = "
            f"{total_unique}"
        )


# ============================================================
# 5. MODEL ARTIFACTS
# ============================================================

section("5. SAVED MODEL CHECK")


required_models = [
    "independent_diabetes.joblib",
    "independent_cvd.joblib",
    "independent_hypertension.joblib",
    "independent_thresholds.json",
    "best_classifier_chain.joblib",
    "best_chain_order.json",
    "federated_classifier_chain.joblib",
    "federated_thresholds.json",
    "fair_federated_classifier_chain.joblib",
    "fair_federated_thresholds.json",
    "shap_backgrounds.joblib",
]


for filename in required_models:

    path = MODEL_DIR / filename

    if path.exists():

        pass_check(
            f"models/{filename}"
        )

    else:

        fail_check(
            f"Missing models/{filename}"
        )


# ============================================================
# 6. FINAL FEDFAIRCHAIN CONTRACT
# ============================================================

section("6. FINAL FEDFAIRCHAIN MODEL CONTRACT")


fair_model_path = (
    MODEL_DIR /
    "fair_federated_classifier_chain.joblib"
)


if fair_model_path.exists():

    bundle = joblib.load(
        fair_model_path
    )


    expected_order = [
        "DIABETES",
        "CVD",
        "HYPERTENSION",
    ]


    actual_order = list(
        bundle["order"]
    )


    if actual_order == expected_order:

        pass_check(
            "Final chain order = "
            "DIABETES -> CVD -> HYPERTENSION"
        )

    else:

        fail_check(
            f"Unexpected chain order: "
            f"{actual_order}"
        )


    features = list(
        bundle["features"]
    )


    if len(features) == 15:

        pass_check(
            "Final predictor count = 15"
        )

    else:

        fail_check(
            f"Predictor count = {len(features)}"
        )


    if "SEX" not in features:

        pass_check(
            "SEX excluded from prediction features"
        )

    else:

        fail_check(
            "SEX found inside prediction features"
        )


    expected_targets = {
        "DIABETES",
        "CVD",
        "HYPERTENSION",
    }


    if set(
        bundle["targets"]
    ) == expected_targets:

        pass_check(
            "All 3 final disease targets present"
        )

    else:

        fail_check(
            "Target contract mismatch"
        )


    thresholds = bundle[
        "thresholds"
    ]


    for target in expected_order:

        threshold = float(
            thresholds[target]
        )

        if 0 <= threshold <= 1:

            pass_check(
                f"{target} threshold = "
                f"{threshold:.2f}"
            )

        else:

            fail_check(
                f"Invalid threshold for "
                f"{target}: {threshold}"
            )


    if (
        bundle.get(
            "protected_attribute"
        )
        == "SEX"
    ):

        pass_check(
            "Protected attribute metadata = SEX"
        )

    else:

        warn_check(
            "Protected attribute metadata "
            "not found or unexpected"
        )


    print(
        "\nFinal prediction features:"
    )

    for index, feature in enumerate(
        features,
        start=1,
    ):

        print(
            f"  {index:02d}. {feature}"
        )


# ============================================================
# 7. FEDERATED CLIENT CHECK
# ============================================================

section("7. FEDERATED CLIENT CHECK")


client_dir = (
    DATA_DIR /
    "federated_clients"
)


client_frames = []


for client_id in range(
    1,
    5,
):

    path = (
        client_dir /
        f"hospital_{client_id}.csv"
    )


    if path.exists():

        df = pd.read_csv(
            path
        )

        client_frames.append(
            df
        )

        pass_check(
            f"Hospital {client_id}: "
            f"{len(df)} patients"
        )

    else:

        fail_check(
            f"Missing Hospital {client_id}"
        )


if len(client_frames) == 4:

    combined_clients = pd.concat(
        client_frames,
        ignore_index=True,
    )


    if len(
        combined_clients
    ) == 6056:

        pass_check(
            "Federated clients contain "
            "all 6,056 training participants"
        )

    else:

        fail_check(
            "Federated client total does "
            "not equal training size"
        )


    if (
        combined_clients[
            "SEQN"
        ].nunique()
        ==
        len(
            combined_clients
        )
    ):

        pass_check(
            "No participant duplication "
            "between hospital nodes"
        )

    else:

        fail_check(
            "Duplicate participants found "
            "across federated clients"
        )


# ============================================================
# 8. FINAL RESULT FILES
# ============================================================

section("8. FINAL RESULT CHECK")


required_results = [
    "final_test_per_label_metrics.csv",
    "final_test_model_comparison.csv",
    "final_test_fairness_group_metrics.csv",
    "final_test_fairness_gaps.csv",
    "final_test_fairness_summary.csv",
    "final_test_m3_vs_m4_fairness.csv",
    "final_test_diagnosis_aware_summary.csv",
    "final_test_diagnosis_aware_per_target.csv",
    "shap_global_importance_all_targets.csv",
]


for filename in required_results:

    path = (
        RESULTS_DIR /
        filename
    )

    if path.exists():

        pass_check(
            f"results/{filename}"
        )

    else:

        fail_check(
            f"Missing result: {filename}"
        )


# ============================================================
# 9. FINAL FIGURES
# ============================================================

section("9. FINAL FIGURE CHECK")


required_figures = [
    "01_final_model_performance.png",
    "02_fedfairchain_per_disease.png",
    "03_fairness_m3_vs_m4.png",
    "04_equal_opportunity_by_disease.png",
    "05_federated_training_convergence.png",
    "06_diagnosis_aware_improvement.png",
    "07_exact_match_accuracy.png",
    "08_hamming_loss.png",
]


for filename in required_figures:

    path = (
        FIGURE_DIR /
        filename
    )

    if path.exists():

        pass_check(
            f"final_figures/{filename}"
        )

    else:

        fail_check(
            f"Missing final figure: {filename}"
        )


shap_figures = [
    "shap_diabetes_global.png",
    "shap_cvd_global.png",
    "shap_hypertension_global.png",
]


for filename in shap_figures:

    path = (
        RESULTS_DIR /
        filename
    )

    if path.exists():

        pass_check(
            filename
        )

    else:

        fail_check(
            f"Missing SHAP figure: {filename}"
        )


# ============================================================
# 10. PREDICTION ENGINE SMOKE TEST
# ============================================================

section("10. FINAL PREDICTION ENGINE SMOKE TEST")


try:

    sys.path.insert(
        0,
        str(ROOT)
    )

    from app.predictor import (
        predict_patient,
    )


    sample_patient = {

        "AGE": 47,
        "BMI": 28.6,
        "WAIST_CM": 96.4,

        "SBP_MEAN": 136,
        "DBP_MEAN": 84,

        "HBA1C": 6.2,

        "TOTAL_CHOLESTEROL": 204,
        "HDL_CHOLESTEROL": 48,

        "CREATININE": 0.92,
        "BUN": 15,
        "ALBUMIN": 4.2,
        "URIC_ACID": 5.8,

        "SERUM_GLUCOSE": 118,

        "SMOKING_STATUS": 1,
        "PHYSICAL_ACTIVITY_ANY": 1,
    }


    unknown_diagnoses = {

        "DIABETES":
            "unknown",

        "CVD":
            "unknown",

        "HYPERTENSION":
            "unknown",
    }


    output = predict_patient(
        sample_patient,
        unknown_diagnoses,
    )


    if output[
        "chain_order"
    ] == [
        "DIABETES",
        "CVD",
        "HYPERTENSION",
    ]:

        pass_check(
            "Prediction engine uses "
            "correct chain order"
        )

    else:

        fail_check(
            "Prediction engine chain mismatch"
        )


    for target in [
        "DIABETES",
        "CVD",
        "HYPERTENSION",
    ]:

        result = (
            output[
                "results"
            ][target]
        )

        probability = (
            result[
                "probability"
            ]
        )


        if (
            probability is not None
            and
            0 <= probability <= 1
        ):

            pass_check(
                f"{target} probability valid: "
                f"{probability:.4f}"
            )

        else:

            fail_check(
                f"Invalid probability for "
                f"{target}"
            )


except Exception as exc:

    fail_check(
        f"Prediction engine failed: {exc}"
    )


# ============================================================
# 11. DIAGNOSIS-AWARE SMOKE TEST
# ============================================================

section("11. DIAGNOSIS-AWARE SMOKE TEST")


try:

    known_diabetes = {

        "DIABETES":
            "yes",

        "CVD":
            "unknown",

        "HYPERTENSION":
            "unknown",
    }


    da_output = predict_patient(
        sample_patient,
        known_diabetes,
    )


    diabetes_result = (
        da_output[
            "results"
        ][
            "DIABETES"
        ]
    )


    if (
        diabetes_result[
            "source"
        ]
        == "known"
        and
        diabetes_result[
            "probability"
        ]
        is None
    ):

        pass_check(
            "Known Diabetes correctly "
            "bypasses prediction"
        )

    else:

        fail_check(
            "Known Diabetes override failed"
        )


    cvd_context = (
        da_output[
            "results"
        ][
            "CVD"
        ][
            "context_used"
        ]
    )


    if (
        len(cvd_context) >= 1
        and
        cvd_context[0][
            "target"
        ]
        == "DIABETES"
        and
        cvd_context[0][
            "value"
        ]
        == 1
        and
        cvd_context[0][
            "source"
        ]
        == "known"
    ):

        pass_check(
            "Known Diabetes is passed "
            "forward into CVD"
        )

    else:

        fail_check(
            "Diagnosis-aware chain context failed"
        )


except Exception as exc:

    fail_check(
        f"Diagnosis-aware test failed: {exc}"
    )


# ============================================================
# 12. FASTAPI IMPORT / ROUTE CHECK
# ============================================================

section("12. FASTAPI CHECK")


try:

    from app.main import app


    route_paths = {
        route.path
        for route in app.routes
    }


    for path in [
        "/",
        "/health",
        "/predict",
        "/docs",
    ]:

        if path in route_paths:

            pass_check(
                f"FastAPI route available: {path}"
            )

        else:

            fail_check(
                f"Missing FastAPI route: {path}"
            )


except Exception as exc:

    fail_check(
        f"FastAPI import failed: {exc}"
    )


# ============================================================
# 13. FINAL MODEL RESULTS SANITY CHECK
# ============================================================

section("13. FINAL RESEARCH RESULT SANITY CHECK")


comparison_file = (
    RESULTS_DIR /
    "final_test_model_comparison.csv"
)


if comparison_file.exists():

    comparison_df = pd.read_csv(
        comparison_file
    )


    expected_models = {
        "M1",
        "M2",
        "M3",
        "M4",
    }


    if set(
        comparison_df[
            "model_id"
        ]
    ) == expected_models:

        pass_check(
            "Final comparison contains M1-M4"
        )

    else:

        fail_check(
            "Final model comparison incomplete"
        )


    m4 = (
        comparison_df[
            comparison_df[
                "model_id"
            ] == "M4"
        ]
        .iloc[0]
    )


    print(
        "\nFrozen M4 FedFairChain test results:"
    )

    print(
        f"  Macro ROC-AUC : "
        f"{m4['macro_roc_auc']:.4f}"
    )

    print(
        f"  Macro PR-AUC  : "
        f"{m4['macro_pr_auc']:.4f}"
    )

    print(
        f"  Macro F1      : "
        f"{m4['macro_f1']:.4f}"
    )

    print(
        f"  Micro F1      : "
        f"{m4['micro_f1']:.4f}"
    )

    print(
        f"  Hamming Loss  : "
        f"{m4['hamming_loss']:.4f}"
    )

    print(
        f"  Exact Match   : "
        f"{m4['exact_match_accuracy']:.4f}"
    )


# ============================================================
# FINAL AUDIT SUMMARY
# ============================================================

section("FINAL AUDIT SUMMARY")


print(
    f"Passed   : {passed}"
)

print(
    f"Warnings : {warnings}"
)

print(
    f"Failed   : {failed}"
)


if failed == 0:

    print(
        "\n"
        "=============================================="
    )

    print(
        "FEDFAIRCHAIN IMPLEMENTATION AUDIT: PASSED"
    )

    print(
        "=============================================="
    )

    print(
        "\nThe core implementation, model artifacts, "
        "evaluation outputs and application are present."
    )

else:

    print(
        "\n"
        "=============================================="
    )

    print(
        "FEDFAIRCHAIN IMPLEMENTATION AUDIT: FAILED"
    )

    print(
        "=============================================="
    )

    print(
        "\nFix the failed checks before final project freeze."
    )


print(
    "\nNo models were trained or modified "
    "during this audit."
)