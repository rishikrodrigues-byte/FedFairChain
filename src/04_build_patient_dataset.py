from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_columns(filename, columns):
    """
    Load only required columns from one NHANES XPT file.
    Also verifies that SEQN is unique.
    """

    path = RAW_DIR / filename

    df = pd.read_sas(path, format="xport")

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {missing}"
        )

    df = df[columns].copy()

    if df["SEQN"].duplicated().any():
        raise ValueError(
            f"Duplicate SEQN values found in {filename}"
        )

    return df


def binary_yes_no(series):
    """
    NHANES questionnaire coding:
        1 = Yes
        2 = No

    Refused / Don't know / other responses become NaN.
    """

    return series.map({
        1.0: 1.0,
        2.0: 0.0,
    })


def combine_binary_conditions(dataframe):
    """
    Composite logic:

    Any component YES  -> 1
    All components NO  -> 0
    Otherwise          -> NaN

    This prevents unknown answers from being treated as healthy.
    """

    any_yes = dataframe.eq(1.0).any(axis=1)
    all_no = dataframe.eq(0.0).all(axis=1)

    result = pd.Series(
        np.nan,
        index=dataframe.index,
        dtype=float
    )

    result.loc[any_yes] = 1.0
    result.loc[all_no] = 0.0

    return result


# ============================================================
# 1. LOAD DEMOGRAPHICS
# ============================================================

demo = load_columns(
    "P_DEMO.xpt",
    [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
    ]
)

print("=" * 72)
print("FEDFAIRCHAIN - PATIENT DATASET CONSTRUCTION")
print("=" * 72)

print(f"\nOriginal NHANES participants: {len(demo)}")


# ============================================================
# 2. KEEP ADULTS AGE >= 20
# ============================================================

patients = demo[
    demo["RIDAGEYR"] >= 20
].copy()

print(f"Adults age 20+: {len(patients)}")


# ============================================================
# 3. LOAD CLINICAL COMPONENTS
# ============================================================

bmx = load_columns(
    "P_BMX.xpt",
    [
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
    ]
)

bp = load_columns(
    "P_BPXO.xpt",
    [
        "SEQN",
        "BPXOSY1",
        "BPXOSY2",
        "BPXOSY3",
        "BPXODI1",
        "BPXODI2",
        "BPXODI3",
    ]
)

ghb = load_columns(
    "P_GHB.xpt",
    [
        "SEQN",
        "LBXGH",
    ]
)

tchol = load_columns(
    "P_TCHOL.xpt",
    [
        "SEQN",
        "LBXTC",
    ]
)

hdl = load_columns(
    "P_HDL.xpt",
    [
        "SEQN",
        "LBDHDD",
    ]
)

biopro = load_columns(
    "P_BIOPRO.xpt",
    [
        "SEQN",
        "LBXSCR",
        "LBXSBU",
        "LBXSAL",
        "LBXSUA",
        "LBXSGL",
    ]
)

diq = load_columns(
    "P_DIQ.xpt",
    [
        "SEQN",
        "DIQ010",
    ]
)

mcq = load_columns(
    "P_MCQ.xpt",
    [
        "SEQN",
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",
    ]
)

kidney = load_columns(
    "P_KIQ_U.xpt",
    [
        "SEQN",
        "KIQ022",
    ]
)

smoking = load_columns(
    "P_SMQ.xpt",
    [
        "SEQN",
        "SMQ020",
        "SMQ040",
    ]
)

activity = load_columns(
    "P_PAQ.xpt",
    [
        "SEQN",
        "PAQ605",
        "PAQ620",
        "PAQ650",
        "PAQ665",
    ]
)


# ============================================================
# 4. MERGE EVERYTHING USING SEQN
# ============================================================

datasets = [
    bmx,
    bp,
    ghb,
    tchol,
    hdl,
    biopro,
    diq,
    mcq,
    kidney,
    smoking,
    activity,
]

for dataset in datasets:

    patients = patients.merge(
        dataset,
        on="SEQN",
        how="left",
        validate="one_to_one",
    )

print(f"Rows after merging: {len(patients)}")

if patients["SEQN"].duplicated().any():
    raise ValueError("Duplicate patients appeared after merging.")

print("SEQN uniqueness check: PASSED")


# ============================================================
# 5. BLOOD PRESSURE
# ============================================================

systolic_columns = [
    "BPXOSY1",
    "BPXOSY2",
    "BPXOSY3",
]

diastolic_columns = [
    "BPXODI1",
    "BPXODI2",
    "BPXODI3",
]

patients["SBP_COUNT"] = (
    patients[systolic_columns]
    .notna()
    .sum(axis=1)
)

patients["DBP_COUNT"] = (
    patients[diastolic_columns]
    .notna()
    .sum(axis=1)
)

patients["SBP_MEAN"] = (
    patients[systolic_columns]
    .mean(axis=1)
)

patients["DBP_MEAN"] = (
    patients[diastolic_columns]
    .mean(axis=1)
)


# ============================================================
# 6. DIABETES LABEL
# ============================================================

# DIQ010:
# 1 = Yes
# 2 = No
# 3 = Borderline
# 9 = Don't know
#
# Borderline and Don't know are NOT forced into either class.

patients["DIABETES"] = (
    patients["DIQ010"]
    .map({
        1.0: 1.0,
        2.0: 0.0,
    })
)


# ============================================================
# 7. CARDIOVASCULAR DISEASE COMPOSITE LABEL
# ============================================================

cvd_raw_columns = [
    "MCQ160B",   # Congestive heart failure
    "MCQ160C",   # Coronary heart disease
    "MCQ160D",   # Angina
    "MCQ160E",   # Heart attack
    "MCQ160F",   # Stroke
]

cvd_binary = patients[
    cvd_raw_columns
].apply(binary_yes_no)

patients["CVD"] = combine_binary_conditions(
    cvd_binary
)


# ============================================================
# 8. KIDNEY DISEASE LABEL
# ============================================================

patients["KIDNEY"] = binary_yes_no(
    patients["KIQ022"]
)


# ============================================================
# 9. SMOKING STATUS
# ============================================================

# Final coding:
#
# 0 = Never smoker
# 1 = Former smoker
# 2 = Current smoker
#
# SMQ020:
# 1 = smoked >=100 cigarettes
# 2 = did not
#
# SMQ040:
# 1 = every day
# 2 = some days
# 3 = not at all

patients["SMOKING_STATUS"] = np.nan

# Never smoker
patients.loc[
    patients["SMQ020"] == 2.0,
    "SMOKING_STATUS"
] = 0.0

# Former smoker
patients.loc[
    (patients["SMQ020"] == 1.0)
    & (patients["SMQ040"] == 3.0),
    "SMOKING_STATUS"
] = 1.0

# Current smoker
patients.loc[
    (patients["SMQ020"] == 1.0)
    & (patients["SMQ040"].isin([1.0, 2.0])),
    "SMOKING_STATUS"
] = 2.0


# ============================================================
# 10. PHYSICAL ACTIVITY
# ============================================================

activity_columns = [
    "PAQ605",
    "PAQ620",
    "PAQ650",
    "PAQ665",
]

activity_binary = patients[
    activity_columns
].apply(binary_yes_no)

patients["PHYSICAL_ACTIVITY_ANY"] = (
    combine_binary_conditions(
        activity_binary
    )
)


# ============================================================
# 11. RENAME FINAL CLINICAL FEATURES
# ============================================================

patients = patients.rename(
    columns={
        "RIDAGEYR": "AGE",
        "RIAGENDR": "SEX",
        "BMXBMI": "BMI",
        "BMXWAIST": "WAIST_CM",
        "LBXGH": "HBA1C",
        "LBXTC": "TOTAL_CHOLESTEROL",
        "LBDHDD": "HDL_CHOLESTEROL",
        "LBXSCR": "CREATININE",
        "LBXSBU": "BUN",
        "LBXSAL": "ALBUMIN",
        "LBXSUA": "URIC_ACID",
        "LBXSGL": "SERUM_GLUCOSE",
    }
)


# ============================================================
# 12. SELECT MASTER COLUMNS
# ============================================================

master_columns = [
    "SEQN",

    # Demographics
    "AGE",
    "SEX",

    # Body measurements
    "BMI",
    "WAIST_CM",

    # Blood pressure
    "SBP_MEAN",
    "DBP_MEAN",
    "SBP_COUNT",
    "DBP_COUNT",

    # Laboratory measurements
    "HBA1C",
    "TOTAL_CHOLESTEROL",
    "HDL_CHOLESTEROL",
    "CREATININE",
    "BUN",
    "ALBUMIN",
    "URIC_ACID",
    "SERUM_GLUCOSE",

    # Lifestyle
    "SMOKING_STATUS",
    "PHYSICAL_ACTIVITY_ANY",

    # Targets
    "DIABETES",
    "CVD",
    "KIDNEY",
]

master = patients[
    master_columns
].copy()


# ============================================================
# 13. SAVE MASTER DATASET
# ============================================================

master_path = (
    PROCESSED_DIR
    / "fedfairchain_master.csv"
)

master.to_csv(
    master_path,
    index=False
)

print(
    f"\nMaster dataset saved to:\n{master_path}"
)


# ============================================================
# 14. COMPLETE-LABEL DATASET
# ============================================================

target_columns = [
    "DIABETES",
    "CVD",
    "KIDNEY",
]

complete_labels = master.dropna(
    subset=target_columns
).copy()

complete_path = (
    PROCESSED_DIR
    / "fedfairchain_complete_labels.csv"
)

complete_labels.to_csv(
    complete_path,
    index=False
)

print(
    f"\nRows with all 3 labels available: "
    f"{len(complete_labels)}"
)


# ============================================================
# 15. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 72)
print("LABEL DISTRIBUTION")
print("=" * 72)

label_report = []

for target in target_columns:

    positive = int(
        complete_labels[target].sum()
    )

    total = int(
        complete_labels[target].notna().sum()
    )

    negative = total - positive

    prevalence = (
        positive / total * 100
        if total > 0
        else 0
    )

    print(
        f"{target:<12} "
        f"Positive={positive:<6} "
        f"Negative={negative:<6} "
        f"Prevalence={prevalence:.2f}%"
    )

    label_report.append({
        "label": target,
        "positive": positive,
        "negative": negative,
        "prevalence_percent": prevalence,
    })


pd.DataFrame(
    label_report
).to_csv(
    RESULTS_DIR / "stage1_label_distribution.csv",
    index=False
)


# ============================================================
# 16. FEATURE MISSINGNESS REPORT
# ============================================================

feature_columns = [
    "AGE",
    "SEX",
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

missingness = (
    complete_labels[feature_columns]
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename("missing_percent")
    .reset_index()
    .rename(columns={"index": "feature"})
)

missingness.to_csv(
    RESULTS_DIR / "stage1_missingness.csv",
    index=False
)

print("\n" + "=" * 72)
print("FEATURE MISSINGNESS")
print("=" * 72)

print(
    missingness.to_string(
        index=False,
        formatters={
            "missing_percent":
                lambda x: f"{x:.2f}%"
        }
    )
)


# ============================================================
# 17. BASIC INTEGRITY CHECKS
# ============================================================

print("\n" + "=" * 72)
print("INTEGRITY CHECKS")
print("=" * 72)

print(
    "Duplicate SEQN:",
    master["SEQN"].duplicated().sum()
)

print(
    "Age below 20:",
    (master["AGE"] < 20).sum()
)

print(
    "Invalid diabetes labels:",
    (~complete_labels["DIABETES"].isin([0, 1])).sum()
)

print(
    "Invalid CVD labels:",
    (~complete_labels["CVD"].isin([0, 1])).sum()
)

print(
    "Invalid kidney labels:",
    (~complete_labels["KIDNEY"].isin([0, 1])).sum()
)


print("\n" + "=" * 72)
print("PATIENT DATASET CONSTRUCTION COMPLETE.")
print("=" * 72)