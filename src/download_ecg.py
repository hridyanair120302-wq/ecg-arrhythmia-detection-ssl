
import pandas as pd
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
OUTPUT = Path("data/raw/ecg_100hz")

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/raw/ptbxl_database.csv")
records = df["filename_lr"].dropna().head(500).tolist()

def download_file(record_path, extension):
    relative_path = f"{record_path}.{extension}"
    destination = OUTPUT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return True

    url = BASE_URL + relative_path
    urllib.request.urlretrieve(url, destination)
    return True

tasks = [
    (record, ext)
    for record in records
    for ext in ("hea", "dat")
]

successful = 0
failed = []

print(f"Downloading {len(records)} ECG recordings...")

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(download_file, record, ext): (record, ext)
        for record, ext in tasks
    }

    for future in tqdm(as_completed(futures), total=len(tasks)):
        record, ext = futures[future]
        try:
            future.result()
            if ext == "dat":
                successful += 1
        except Exception as e:
            failed.append((record, ext, str(e)))

print(f"\nSuccessful ECG recordings: {successful}")
print(f"Failed files: {len(failed)}")

if failed:
    print("First failures:", failed[:5])
