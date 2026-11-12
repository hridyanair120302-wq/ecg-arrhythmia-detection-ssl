
import pandas as pd
import urllib.request
import time
from pathlib import Path

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
OUTPUT = Path("data/raw/ecg_100hz")

df = pd.read_csv("data/processed/ecg_labels.csv")

failed = []

for record in df["filename_lr"]:
    for ext in ["hea", "dat"]:
        path = f"{record}.{ext}"
        destination = OUTPUT / path

        if destination.exists() and destination.stat().st_size > 0:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(5):
            try:
                urllib.request.urlretrieve(BASE + path, destination)
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            failed.append(path)

print("Still missing:", failed)
print("Retry complete!")
