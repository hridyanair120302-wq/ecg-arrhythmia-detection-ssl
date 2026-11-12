import pandas as pd
import ast

df = pd.read_csv("data/raw/ptbxl_database.csv")

print("Total ECG records:", len(df))
print("\nAvailable sampling frequencies:")
print("100 Hz:", df["filename_lr"].notna().sum())
print("500 Hz:", df["filename_hr"].notna().sum())

df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

print("\nMost common diagnostic codes:")

counts = {}

for codes in df["scp_codes"]:
    for code, value in codes.items():
        counts[code] = counts.get(code, 0) + 1

for code, count in sorted(
    counts.items(), key=lambda x: x[1], reverse=True
)[:15]:
    print(code, count)

print("\nRecommended patient-level folds:")
print(df["strat_fold"].value_counts().sort_index())
