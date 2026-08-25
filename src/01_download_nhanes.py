from pathlib import Path
import requests

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

FILES = [
    "P_DEMO",
    "P_BMX",
    "P_BPXO",
    "P_GHB",
    "P_TCHOL",
    "P_HDL",
    "P_BIOPRO",
    "P_DIQ",
    "P_MCQ",
    "P_KIQ_U",
    "P_SMQ",
    "P_PAQ",
    "P_BPQ",
]

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_file(name):
    filename = f"{name}.xpt"
    url = f"{BASE_URL}/{filename}"
    destination = RAW_DIR / filename

    if destination.exists():
        print(f"[SKIP] {filename} already exists")
        return

    print(f"[DOWNLOAD] {filename}")

    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    response.raise_for_status()
    destination.write_bytes(response.content)

    size_mb = destination.stat().st_size / (1024 * 1024)

    print(f"[OK] {filename} - {size_mb:.2f} MB")


print("=" * 60)
print("FEDFAIRCHAIN - NHANES DOWNLOAD")
print("=" * 60)

for file_name in FILES:
    try:
        download_file(file_name)
    except Exception as error:
        print(f"[ERROR] {file_name}")
        print(error)

print("\nDownload process finished.")