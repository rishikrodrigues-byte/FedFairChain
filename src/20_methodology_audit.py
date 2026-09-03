from __future__ import annotations

from pathlib import Path
import json
import re

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
MODEL_PATH = ROOT / "models" / "fair_federated_classifier_chain.joblib"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BUILD_SCRIPT = SRC_DIR / "04_build_patient_dataset.py"
CHAIN_SCRIPTS = {
    "centralized_classifier_chain": SRC_DIR / "08_compare_classifier_chains.py",
    "federated_classifier_chain": SRC_DIR / "11_train_federated_chain.py",
    "fair_federated_classifier_chain": SRC_DIR / "13_train_fair_federated_chain.py",
}

TARGET_REPORT = RESULTS_DIR / "target_provenance_audit.csv"
CHAIN_REPORT = RESULTS_DIR / "chain_training_context_audit.json"


def require_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def require_tokens(text: str, tokens: list[str], context: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise RuntimeError(f"{context}: expected source tokens missing: {missing}")


def extract_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Could not locate section marker: {start_marker}")
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        raise RuntimeError(f"Could not locate section end marker: {end_marker}")
    return text[start:end]


def main() -> None:
    print("=" * 88)
    print("FEDFAIRCHAIN - METHODOLOGY AUDIT (VALIDATION/STATIC AUDIT ONLY)")
    print("=" * 88)
    print("[PASS] This audit does not load data/processed/test.csv")

    build_text = require_text(BUILD_SCRIPT)
    bundle = joblib.load(MODEL_PATH)
    features = list(bundle["features"])
    order = list(bundle["order"])

    diabetes_section = extract_section(
        build_text,
        "# 6. DIABETES LABEL",
        "# 7. CARDIOVASCULAR DISEASE COMPOSITE LABEL",
    )
    cvd_section = extract_section(
        build_text,
        "# 7. CARDIOVASCULAR DISEASE COMPOSITE LABEL",
        "# 8. KIDNEY DISEASE LABEL",
    )
    hypertension_section = extract_section(
        build_text,
        "# HYPERTENSION LABEL",
        "# 9. SMOKING STATUS",
    )

    # These variables are taken directly from the existing build script and
    # are verified below before being reported.
    provenance = {
        "DIABETES": {
            "vars": ["DIQ010"],
            "definition": "DIQ010: 1 -> positive; 2 -> negative; borderline/don't-know/other -> missing.",
            "section": diabetes_section,
            "notes": "Questionnaire-defined target. HBA1C and SERUM_GLUCOSE are predictors but are not referenced in the target formula.",
        },
        "CVD": {
            "vars": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
            "definition": "Composite questionnaire target: any component Yes -> positive; all components No -> negative; otherwise -> missing.",
            "section": cvd_section,
            "notes": "Composite target uses the five questionnaire diagnosis/history fields only.",
        },
        "HYPERTENSION": {
            "vars": ["BPQ020", "BPQ030", "BPQ050A"],
            "definition": "BPQ020==2 -> negative; BPQ030==1 or BPQ050A==1 -> positive; other uncertain cases -> missing.",
            "section": hypertension_section,
            "notes": "Questionnaire/treatment-defined target. SBP_MEAN and DBP_MEAN are predictors but do not directly determine this target.",
        },
    }

    rows = []
    any_direct_leakage = False

    for target, info in provenance.items():
        require_tokens(info["section"], info["vars"], f"{target} provenance")

        # Direct target leakage is defined narrowly here: a model predictor or
        # its constructed clinical name is directly referenced in the target
        # assignment logic. Proximal biomarkers are reported separately.
        predictor_overlap = sorted(set(info["vars"]).intersection(features))

        sensitive_predictor_tokens = {
            "DIABETES": ["HBA1C", "SERUM_GLUCOSE", "LBXGH", "LBXSGL"],
            "CVD": [],
            "HYPERTENSION": [
                "SBP_MEAN", "DBP_MEAN", "BPXOSY1", "BPXOSY2", "BPXOSY3",
                "BPXODI1", "BPXODI2", "BPXODI3",
            ],
        }[target]
        formula_predictor_tokens = [
            token for token in sensitive_predictor_tokens if token in info["section"]
        ]
        direct = bool(predictor_overlap or formula_predictor_tokens)
        any_direct_leakage = any_direct_leakage or direct

        rows.append(
            {
                "target": target,
                "label_source_variables": ";".join(info["vars"]),
                "label_definition": info["definition"],
                "predictor_overlap": ";".join(predictor_overlap + formula_predictor_tokens),
                "direct_leakage_detected": direct,
                "notes": info["notes"],
            }
        )

    pd.DataFrame(rows).to_csv(TARGET_REPORT, index=False)

    print("\nA1. TARGET PROVENANCE / DIRECT LEAKAGE")
    for row in rows:
        status = "REVIEW REQUIRED" if row["direct_leakage_detected"] else "PASS"
        print(f"[{status}] {row['target']}: {row['label_source_variables']}")
        print(f"  {row['label_definition']}")
        print(f"  {row['notes']}")

    chain_audit = {
        "test_dataset_loaded": False,
        "production_chain_order": order,
        "scripts": {},
        "same_row_prediction_context_leakage_detected": False,
        "oof_correction_required_by_a2": False,
        "methodological_note": (
            "All inspected training scripts use teacher forcing: the ground-truth previous label is appended "
            "to downstream training stages. This is the standard hard-label classifier-chain training pattern "
            "and is not same-row prediction leakage. Inference instead appends previous hard predictions (or "
            "user-supplied known diagnoses), so a train/inference context mismatch remains and should be examined "
            "by the planned hard-label versus probability-context validation ablation."
        ),
    }

    for name, path in CHAIN_SCRIPTS.items():
        text = require_text(path)
        teacher_forcing_markers = [
            "true previous label",
            "Teacher forcing",
        ]
        uses_ground_truth = any(marker.lower() in text.lower() for marker in teacher_forcing_markers)
        appends_y = bool(re.search(r"np\.column_stack\s*\([\s\S]{0,300}?\by\b", text)) or "y_train" in text
        if not (uses_ground_truth and appends_y):
            raise RuntimeError(f"Could not verify teacher-forcing context in {path.name}")

        chain_audit["scripts"][name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "training_context": "ground_truth_previous_labels",
            "classification": "A",
            "same_row_model_predictions_used_for_training_context": False,
            "out_of_fold_predictions_used": False,
        }

    CHAIN_REPORT.write_text(json.dumps(chain_audit, indent=2), encoding="utf-8")

    print("\nA2. CLASSIFIER-CHAIN TRAINING CONTEXT")
    for name, info in chain_audit["scripts"].items():
        print(f"[PASS] {name}: {info['training_context']} (classification {info['classification']})")
    print("[PASS] Same-row model predictions are NOT used as downstream training context.")
    print("[PASS] OOF correction is NOT required by the A2 trigger.")
    print("[NOTE] Teacher-forced training vs predicted-context inference remains an ablation question, not leakage.")

    if any_direct_leakage:
        overall = "REVIEW REQUIRED"
    else:
        overall = "PASS"

    print("\n" + "=" * 88)
    print(f"METHODOLOGY AUDIT RESULT: {overall}")
    print("=" * 88)
    print(f"Target provenance report: {TARGET_REPORT.relative_to(ROOT)}")
    print(f"Chain context report     : {CHAIN_REPORT.relative_to(ROOT)}")

    if any_direct_leakage:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
