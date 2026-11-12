
import pandas as pd
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
OUTPUT = Path("data/raw/ecg_100hz")

df = pd.read_csv("data/processed/ecg_labels.csv")

tasks = []

for record_path in df["filename_lr"]:
    for ext in ["hea", "dat"]:
        tasks.append(f"{record_path}.{ext}")

def download(path):
    destination = OUTPUT / path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return True

    urllib.request.urlretrieve(BASE_URL + path, destination)
    return True

failed = []

print(f"Downloading {len(df)} selected ECG recordings...")

with ThreadPoolExecutor(max_workers=12) as executor:
    futures = {executor.submit(download, p): p for p in tasks}

    for future in tqdm(as_completed(futures), total=len(tasks)):
        try:
            future.result()
        except Exception as e:
            failed.append((futures[future], str(e)))

print("\nRecordings:", len(df))
print("Files downloaded:", len(tasks) - len(failed))
print("Failed files:", len(failed))

if failed:
    print(failed[:10])
