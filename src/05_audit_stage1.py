from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "data/processed/fedfairchain_complete_labels.csv"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

PROTECTED_ATTRIBUTE = "SEX"

TARGETS = [
    "DIABETES",
    "HYPERTENSION",
    "CVD",
]


print("=" * 78)
print("FEDFAIRCHAIN - STAGE 1 FINAL DATA AUDIT")
print("=" * 78)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"\nRows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 2. BASIC INTEGRITY
# ============================================================

print("\n" + "=" * 78)
print("BASIC INTEGRITY")
print("=" * 78)

duplicate_seqn = df["SEQN"].duplicated().sum()

print("Duplicate SEQN:", duplicate_seqn)

for target in TARGETS:

    invalid = (
        ~df[target].isin([0, 1])
    ).sum()

    print(
        f"Invalid {target:<8}: {invalid}"
    )


# ============================================================
# 3. TARGET PREVALENCE
# ============================================================

print("\n" + "=" * 78)
print("TARGET PREVALENCE")
print("=" * 78)

for target in TARGETS:

    positives = int(df[target].sum())

    prevalence = (
        df[target].mean() * 100
    )

    print(
        f"{target:<12} "
        f"positive={positives:<5} "
        f"prevalence={prevalence:.2f}%"
    )


# ============================================================
# 4. MULTI-LABEL COMORBIDITY PATTERNS
# ============================================================

df["LABEL_PATTERN"] = (
    df[TARGETS]
    .astype(int)
    .astype(str)
    .agg("".join, axis=1)
)

pattern_counts = (
    df["LABEL_PATTERN"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 78)
print("MULTI-LABEL COMORBIDITY PATTERNS")
print("=" * 78)

print("\nPattern meaning:")
print("000 = none")
print("001 = CVD only")
print("010 = hypertension only")
print("011 = hypertension + CVD")
print("100 = diabetes only")
print("101 = diabetes + CVD")
print("110 = diabetes + hypertension")
print("111 = diabetes + hypertension + CVD\n")

for pattern, count in pattern_counts.items():

    percentage = (
        count / len(df) * 100
    )

    print(
        f"{pattern}: "
        f"{count:<5} "
        f"({percentage:.2f}%)"
    )


pattern_counts.rename(
    "count"
).to_csv(
    RESULTS_DIR /
    "stage1_label_patterns.csv"
)


# ============================================================
# 5. DISEASE CO-OCCURRENCE
# ============================================================

print("\n" + "=" * 78)
print("DISEASE CO-OCCURRENCE MATRIX")
print("=" * 78)

target_matrix = (
    df[TARGETS]
    .astype(int)
)

cooccurrence = (
    target_matrix.T
    @ target_matrix
)

print(cooccurrence)

cooccurrence.to_csv(
    RESULTS_DIR /
    "stage1_disease_cooccurrence.csv"
)


# ============================================================
# 6. SEX DISTRIBUTION
# ============================================================

print("\n" + "=" * 78)
print("PROTECTED ATTRIBUTE AUDIT - SEX")
print("=" * 78)

sex_counts = (
    df[PROTECTED_ATTRIBUTE]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nParticipant counts:")

for value, count in sex_counts.items():

    label = {
        1.0: "Male",
        2.0: "Female",
    }.get(value, f"Unknown({value})")

    print(
        f"{label:<12}: {count}"
    )


print("\nDisease prevalence by sex:")

for sex_code, sex_name in [
    (1.0, "Male"),
    (2.0, "Female"),
]:

    group = df[
        df[PROTECTED_ATTRIBUTE]
        == sex_code
    ]

    print(
        f"\n{sex_name} "
        f"(n={len(group)})"
    )

    for target in TARGETS:

        prevalence = (
            group[target].mean()
            * 100
        )

        print(
            f"  {target:<10}: "
            f"{prevalence:.2f}%"
        )


# ============================================================
# 7. FEATURE MISSINGNESS
# ============================================================

print("\n" + "=" * 78)
print("FEATURE MISSINGNESS")
print("=" * 78)

missingness = (
    df[FEATURES]
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

for feature, percentage in missingness.items():

    print(
        f"{feature:<24} "
        f"{percentage:.2f}%"
    )


# ============================================================
# 8. COMPLETE CASE IMPACT
# ============================================================

complete_case = (
    df
    .dropna(subset=FEATURES)
)

complete_percentage = (
    len(complete_case)
    / len(df)
    * 100
)

print("\n" + "=" * 78)
print("MISSING-DATA STRATEGY CHECK")
print("=" * 78)

print(
    "Rows if we deleted every row "
    "with any missing predictor:"
)

print(
    f"{len(complete_case)} / {len(df)} "
    f"({complete_percentage:.2f}%)"
)

print(
    "\nDecision: "
    "DO NOT delete all incomplete rows."
)

print(
    "We will use training-only "
    "median imputation instead."
)


# ============================================================
# 9. FEATURE RANGE INSPECTION
# ============================================================

print("\n" + "=" * 78)
print("NUMERIC FEATURE RANGE INSPECTION")
print("=" * 78)

range_report = (
    df[FEATURES]
    .describe()
    .T[
        [
            "count",
            "mean",
            "std",
            "min",
            "50%",
            "max",
        ]
    ]
)

print(
    range_report.round(3)
)

range_report.to_csv(
    RESULTS_DIR /
    "stage1_feature_ranges.csv"
)


# ============================================================
# 10. LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 78)
print("LEAKAGE CHECK")
print("=" * 78)

overlap = (
    set(FEATURES)
    .intersection(TARGETS)
)

print(
    "Target columns present "
    "inside predictor list:",
    overlap
)

print(
    "Protected attribute used "
    "as predictor:",
    PROTECTED_ATTRIBUTE in FEATURES
)


# ============================================================
# 11. FINAL AUTOMATIC CHECK
# ============================================================

checks = {
    "No duplicate patients":
        duplicate_seqn == 0,

    "All targets binary":
        all(
            df[target]
            .isin([0, 1])
            .all()
            for target in TARGETS
        ),

    "Enough patient records":
        len(df) >= 5000,

    "No direct target leakage":
        len(overlap) == 0,

    "Sex excluded from predictors":
        PROTECTED_ATTRIBUTE
        not in FEATURES,

    "Worst feature missingness < 30%":
        missingness.max()
        < 30,
}


print("\n" + "=" * 78)
print("FINAL STAGE 1 CHECKS")
print("=" * 78)

all_passed = True

for check_name, passed in checks.items():

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"[{status}] {check_name}"
    )

    if not passed:
        all_passed = False


print("\n" + "=" * 78)

if all_passed:

    print(
        "STAGE 1 DATA AUDIT PASSED."
    )

else:

    print(
        "STAGE 1 DATA AUDIT FAILED."
    )

print("=" * 78)