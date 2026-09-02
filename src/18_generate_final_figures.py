from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")

FIGURE_DIR = (
    RESULTS_DIR /
    "final_figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DPI = 300


print("=" * 92)
print("FEDFAIRCHAIN - STAGE 8.3 FINAL FIGURE GENERATION")
print("=" * 92)

print(
    "\nNo models are loaded."
)

print(
    "No training, prediction or threshold tuning is performed."
)

print(
    "Only previously saved frozen result CSVs are visualized."
)


# ============================================================
# HELPER
# ============================================================

def save_figure(
    filename,
):

    path = (
        FIGURE_DIR /
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"[SAVED] {path}"
    )


# ============================================================
# FIGURE 1
# FINAL MODEL PERFORMANCE COMPARISON
# ============================================================

comparison_path = (
    RESULTS_DIR /
    "final_test_model_comparison.csv"
)

comparison_df = pd.read_csv(
    comparison_path
)


metrics = [
    "macro_pr_auc",
    "macro_roc_auc",
    "macro_f1",
    "micro_f1",
]

metric_labels = [
    "Macro PR-AUC",
    "Macro ROC-AUC",
    "Macro F1",
    "Micro F1",
]


x = np.arange(
    len(comparison_df)
)

width = 0.18


plt.figure(
    figsize=(11, 6)
)


for index, metric in enumerate(
    metrics
):

    offset = (
        index
        - (len(metrics) - 1) / 2
    ) * width

    plt.bar(
        x + offset,
        comparison_df[metric],
        width,
        label=metric_labels[index],
    )


plt.xticks(
    x,
    comparison_df["model_id"],
)

plt.ylim(
    0,
    1,
)

plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Final Held-Out Test Performance"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "01_final_model_performance.png"
)


# ============================================================
# FIGURE 2
# FEDFAIRCHAIN PER-DISEASE RESULTS
# ============================================================

per_label_path = (
    RESULTS_DIR /
    "final_test_per_label_metrics.csv"
)

per_label_df = pd.read_csv(
    per_label_path
)


m4_df = (
    per_label_df[
        per_label_df[
            "model_id"
        ] == "M4"
    ]
    .copy()
)


target_labels = {
    "DIABETES":
        "Diabetes",

    "HYPERTENSION":
        "Hypertension",

    "CVD":
        "CVD",
}


m4_df[
    "display_target"
] = (
    m4_df["target"]
    .map(target_labels)
)


metrics = [
    "roc_auc",
    "pr_auc",
    "f1",
    "balanced_accuracy",
]

labels = [
    "ROC-AUC",
    "PR-AUC",
    "F1",
    "Balanced Accuracy",
]


x = np.arange(
    len(m4_df)
)

width = 0.18


plt.figure(
    figsize=(10, 6)
)


for index, metric in enumerate(
    metrics
):

    offset = (
        index
        - (len(metrics) - 1) / 2
    ) * width

    plt.bar(
        x + offset,
        m4_df[metric],
        width,
        label=labels[index],
    )


plt.xticks(
    x,
    m4_df[
        "display_target"
    ],
)

plt.ylim(
    0,
    1,
)

plt.ylabel(
    "Score"
)

plt.title(
    "FedFairChain Performance by Disease"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "02_fedfairchain_per_disease.png"
)


# ============================================================
# FIGURE 3
# OVERALL FAIRNESS M3 VS M4
# ============================================================

fairness_path = (
    RESULTS_DIR /
    "final_test_fairness_summary.csv"
)

fairness_df = pd.read_csv(
    fairness_path
)


fairness_df = (
    fairness_df[
        fairness_df[
            "model_id"
        ].isin(
            ["M3", "M4"]
        )
    ]
    .copy()
)


fair_metrics = [
    "mean_abs_dp_gap",
    "mean_abs_eo_gap",
    "mean_abs_fpr_gap",
]

fair_labels = [
    "Demographic Parity",
    "Equal Opportunity",
    "False Positive Rate",
]


x = np.arange(
    len(fair_labels)
)

width = 0.34


plt.figure(
    figsize=(10, 6)
)


for index, model_id in enumerate(
    ["M3", "M4"]
):

    row = (
        fairness_df[
            fairness_df[
                "model_id"
            ] == model_id
        ]
        .iloc[0]
    )

    values = [
        row[metric]
        for metric in fair_metrics
    ]


    plt.bar(
        x
        + (
            index - 0.5
        ) * width,
        values,
        width,
        label=model_id,
    )


plt.xticks(
    x,
    fair_labels,
)

plt.ylabel(
    "Mean Absolute Gap"
)

plt.title(
    "Fairness Comparison: Federated Chain vs FedFairChain"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "03_fairness_m3_vs_m4.png"
)


# ============================================================
# FIGURE 4
# PER-DISEASE EQUAL OPPORTUNITY GAP
# ============================================================

m3_m4_path = (
    RESULTS_DIR /
    "final_test_m3_vs_m4_fairness.csv"
)

m3_m4_df = pd.read_csv(
    m3_m4_path
)


x = np.arange(
    len(m3_m4_df)
)

width = 0.34


plt.figure(
    figsize=(9, 6)
)


plt.bar(
    x - width / 2,
    m3_m4_df[
        "M3_abs_EO"
    ],
    width,
    label=(
        "M3 Federated Chain"
    ),
)


plt.bar(
    x + width / 2,
    m3_m4_df[
        "M4_abs_EO"
    ],
    width,
    label=(
        "M4 FedFairChain"
    ),
)


plt.xticks(
    x,
    [
        target_labels.get(
            value,
            value,
        )
        for value
        in m3_m4_df[
            "target"
        ]
    ],
)

plt.ylabel(
    "Absolute Equal Opportunity Gap"
)

plt.title(
    "Equal Opportunity Gap by Disease"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "04_equal_opportunity_by_disease.png"
)


# ============================================================
# FIGURE 5
# FEDERATED TRAINING CONVERGENCE
# ============================================================

standard_history_path = (
    RESULTS_DIR /
    "federated_training_history.csv"
)

fair_history_path = (
    RESULTS_DIR /
    "fair_federated_training_history.csv"
)


standard_history = pd.read_csv(
    standard_history_path
)

fair_history = pd.read_csv(
    fair_history_path
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    standard_history[
        "round"
    ],
    standard_history[
        "macro_pr_auc"
    ],
    marker="o",
    markersize=3,
    label=(
        "Standard Federated Chain"
    ),
)


plt.plot(
    fair_history[
        "round"
    ],
    fair_history[
        "macro_pr_auc"
    ],
    marker="o",
    markersize=3,
    label=(
        "Fairness-Aware FedFairChain"
    ),
)


plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "Validation Macro PR-AUC"
)

plt.title(
    "Federated Training Convergence"
)

plt.legend()

plt.grid(
    alpha=0.25,
)


save_figure(
    "05_federated_training_convergence.png"
)


# ============================================================
# FIGURE 6
# DIAGNOSIS-AWARE IMPROVEMENT
# ============================================================

diagnosis_path = (
    RESULTS_DIR /
    "final_test_diagnosis_aware_summary.csv"
)

diagnosis_df = pd.read_csv(
    diagnosis_path
)


plt.figure(
    figsize=(11, 7)
)


y = np.arange(
    len(diagnosis_df)
)


plt.barh(
    y,
    diagnosis_df[
        "pr_auc_delta"
    ],
)


plt.yticks(
    y,
    diagnosis_df[
        "scenario"
    ],
)

plt.axvline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Change in PR-AUC"
)

plt.title(
    "Diagnosis-Aware Inference: Held-Out PR-AUC Improvement"
)

plt.grid(
    axis="x",
    alpha=0.25,
)


save_figure(
    "06_diagnosis_aware_improvement.png"
)


# ============================================================
# FIGURE 7
# FINAL MODEL EXACT-MATCH PERFORMANCE
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.bar(
    comparison_df[
        "model_id"
    ],
    comparison_df[
        "exact_match_accuracy"
    ],
)


plt.ylim(
    0,
    1,
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Exact Match Accuracy"
)

plt.title(
    "Multi-Label Exact Match Accuracy"
)

plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "07_exact_match_accuracy.png"
)


# ============================================================
# FIGURE 8
# FINAL MODEL HAMMING LOSS
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.bar(
    comparison_df[
        "model_id"
    ],
    comparison_df[
        "hamming_loss"
    ],
)


plt.xlabel(
    "Model"
)

plt.ylabel(
    "Hamming Loss"
)

plt.title(
    "Multi-Label Hamming Loss"
)

plt.grid(
    axis="y",
    alpha=0.25,
)


save_figure(
    "08_hamming_loss.png"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 92)
print("FINAL FIGURES GENERATED")
print("=" * 92)

for path in sorted(
    FIGURE_DIR.glob(
        "*.png"
    )
):

    print(path)


print("\nExisting SHAP figures:")

for target in [
    "diabetes",
    "cvd",
    "hypertension",
]:

    shap_path = (
        RESULTS_DIR /
        f"shap_{target}_global.png"
    )

    if shap_path.exists():

        print(
            shap_path
        )


print("\n" + "=" * 92)
print("STAGE 8.3 COMPLETE")
print("=" * 92)

print(
    "\nThese figures visualize frozen "
    "validation/test results only."
)

print(
    "No model was modified."
)