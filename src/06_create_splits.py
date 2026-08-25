from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/fedfairchain_complete_labels.csv")

OUTPUT_DIR = Path("data/processed")

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VAL_PATH = OUTPUT_DIR / "validation.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

RANDOM_STATE = 42

TARGETS = [
    "DIABETES",
    "HYPERTENSION",
    "CVD",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 78)
print("FEDFAIRCHAIN - TRAIN / VALIDATION / TEST SPLIT")
print("=" * 78)

df = pd.read_csv(DATA_PATH)

print(f"\nLoaded dataset: {DATA_PATH}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = ["SEQN"] + TARGETS

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["SEQN"].duplicated().any():
    raise ValueError("Duplicate SEQN values detected.")


# ============================================================
# CREATE TEMPORARY MULTI-LABEL PATTERN
# ============================================================

# Example:
# Diabetes=1, Hypertension=0, CVD=1
# becomes "101"

label_pattern = (
    df[TARGETS]
    .astype(int)
    .astype(str)
    .agg("".join, axis=1)
)

print("\nOverall multi-label patterns:")
print(label_pattern.value_counts().sort_index())


# ============================================================
# FIRST SPLIT
# 70% TRAIN, 30% TEMPORARY
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=label_pattern,
)


# ============================================================
# SECOND SPLIT
# Temporary 30% -> 15% VALIDATION + 15% TEST
# ============================================================

temp_pattern = (
    temp_df[TARGETS]
    .astype(int)
    .astype(str)
    .agg("".join, axis=1)
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_pattern,
)


# ============================================================
# RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# SAVE SPLITS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)


# ============================================================
# REPORT FUNCTION
# ============================================================

def report_split(name, split_df):
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)

    print(f"Rows: {len(split_df)}")

    print("\nTarget prevalence:")

    for target in TARGETS:
        positive = int(split_df[target].sum())
        prevalence = split_df[target].mean() * 100

        print(
            f"{target:<12} "
            f"positive={positive:<5} "
            f"prevalence={prevalence:.2f}%"
        )

    pattern = (
        split_df[TARGETS]
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
    )

    print("\nMulti-label patterns:")

    counts = pattern.value_counts().sort_index()

    for pattern_name, count in counts.items():
        percentage = count / len(split_df) * 100

        print(
            f"{pattern_name}: "
            f"{count:<5} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# PRINT REPORTS
# ============================================================

report_split("TRAINING SET", train_df)
report_split("VALIDATION SET", val_df)
report_split("TEST SET", test_df)


# ============================================================
# LEAKAGE / OVERLAP CHECKS
# ============================================================

train_ids = set(train_df["SEQN"])
val_ids = set(val_df["SEQN"])
test_ids = set(test_df["SEQN"])

train_val_overlap = train_ids.intersection(val_ids)
train_test_overlap = train_ids.intersection(test_ids)
val_test_overlap = val_ids.intersection(test_ids)

print("\n" + "=" * 78)
print("SPLIT INTEGRITY CHECKS")
print("=" * 78)

print(
    "Train + Validation + Test:",
    len(train_df) + len(val_df) + len(test_df)
)

print(
    "Original dataset rows:",
    len(df)
)

print(
    "Train/Validation overlap:",
    len(train_val_overlap)
)

print(
    "Train/Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation/Test overlap:",
    len(val_test_overlap)
)


# ============================================================
# FINAL CHECKS
# ============================================================

checks = {
    "All patients preserved":
        len(train_df) + len(val_df) + len(test_df) == len(df),

    "No Train/Validation overlap":
        len(train_val_overlap) == 0,

    "No Train/Test overlap":
        len(train_test_overlap) == 0,

    "No Validation/Test overlap":
        len(val_test_overlap) == 0,
}

print("\nFinal checks:")

all_passed = True

for name, passed in checks.items():

    if passed:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")
        all_passed = False


if not all_passed:
    raise RuntimeError(
        "Dataset split integrity check failed."
    )


print("\nSaved:")
print(TRAIN_PATH)
print(VAL_PATH)
print(TEST_PATH)

print("\n" + "=" * 78)
print("STAGE 2.1 DATA SPLIT COMPLETE.")
print("=" * 78)