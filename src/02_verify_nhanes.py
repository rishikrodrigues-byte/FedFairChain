from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

files = sorted(RAW_DIR.glob("*.xpt"))

print("=" * 72)
print("FEDFAIRCHAIN - NHANES DATA VERIFICATION")
print("=" * 72)

print(f"\nFiles found: {len(files)}\n")

all_valid = True

for file in files:
    try:
        df = pd.read_sas(file, format="xport")

        has_seqn = "SEQN" in df.columns

        print(
            f"[OK] {file.name:<18} "
            f"rows={df.shape[0]:<7} "
            f"columns={df.shape[1]:<5} "
            f"SEQN={has_seqn}"
        )

        if not has_seqn:
            all_valid = False

    except Exception as error:
        all_valid = False
        print(f"[ERROR] {file.name}")
        print(error)

print("\n" + "=" * 72)

if len(files) == 12 and all_valid:
    print("ALL 12 NHANES FILES VERIFIED SUCCESSFULLY.")
else:
    print("NHANES VERIFICATION FAILED.")
    print(f"Expected files: 12")
    print(f"Files found   : {len(files)}")

print("=" * 72)
