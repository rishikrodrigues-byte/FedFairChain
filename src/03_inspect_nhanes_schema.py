from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

EXPECTED = {
    "P_DEMO.xpt": [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
    ],

    "P_BMX.xpt": [
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
    ],

    "P_BPXO.xpt": [
        "SEQN",
        "BPXOSY1",
        "BPXOSY2",
        "BPXOSY3",
        "BPXODI1",
        "BPXODI2",
        "BPXODI3",
    ],

    "P_GHB.xpt": [
        "SEQN",
        "LBXGH",
    ],

    "P_TCHOL.xpt": [
        "SEQN",
        "LBXTC",
    ],

    "P_HDL.xpt": [
        "SEQN",
        "LBDHDD",
    ],

    "P_BIOPRO.xpt": [
        "SEQN",
        "LBXSCR",
        "LBXSBU",
        "LBXSAL",
        "LBXSUA",
        "LBXSGL",
    ],

    "P_DIQ.xpt": [
        "SEQN",
        "DIQ010",
    ],

    "P_MCQ.xpt": [
        "SEQN",
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",
    ],

    "P_KIQ_U.xpt": [
        "SEQN",
        "KIQ022",
    ],

    "P_SMQ.xpt": [
        "SEQN",
        "SMQ020",
        "SMQ040",
    ],

    "P_PAQ.xpt": [
        "SEQN",
        "PAQ605",
        "PAQ620",
        "PAQ650",
        "PAQ665",
    ],
}


print("=" * 78)
print("FEDFAIRCHAIN - NHANES VARIABLE INSPECTION")
print("=" * 78)

everything_found = True

for filename, expected_columns in EXPECTED.items():

    path = RAW_DIR / filename

    print(f"\n--- {filename} ---")

    df = pd.read_sas(path, format="xport")

    for column in expected_columns:

        if column in df.columns:
            non_missing = df[column].notna().sum()

            print(
                f"[FOUND] {column:<10}"
                f" non-missing={non_missing:<7}"
                f" missing={df[column].isna().sum()}"
            )

        else:
            everything_found = False
            print(f"[MISSING] {column}")


print("\n" + "=" * 78)
print("DIAGNOSIS VARIABLE VALUE CHECK")
print("=" * 78)


diagnosis_variables = {
    "P_DIQ.xpt": ["DIQ010"],

    "P_MCQ.xpt": [
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",
    ],

    "P_KIQ_U.xpt": ["KIQ022"],
}


for filename, columns in diagnosis_variables.items():

    df = pd.read_sas(
        RAW_DIR / filename,
        format="xport"
    )

    print(f"\n--- {filename} ---")

    for column in columns:

        if column in df.columns:

            print(f"\n{column}")

            counts = (
                df[column]
                .value_counts(dropna=False)
                .sort_index()
            )

            print(counts)


print("\n" + "=" * 78)

if everything_found:
    print("ALL REQUIRED FEDFAIRCHAIN VARIABLES WERE FOUND.")
else:
    print("WARNING: ONE OR MORE EXPECTED VARIABLES WERE NOT FOUND.")

print("=" * 78)