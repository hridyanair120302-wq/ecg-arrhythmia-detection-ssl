
import pandas as pd
import ast
from collections import Counter

df = pd.read_csv("data/raw/ptbxl_database.csv")
statements = pd.read_csv(
    "data/raw/scp_statements.csv", index_col=0
)

df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

rhythm_codes = statements[statements["rhythm"] == 1]

counter = Counter()

for codes in df["scp_codes"]:
    for code in codes:
        if code in rhythm_codes.index:
            counter[code] += 1

print("\nRHYTHM CLASS DISTRIBUTION\n")

import pandas as pd
import ast

df = pd.read_csv("data/raw/ptbxl_database.csv")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

# Keep only unambiguous AFIB and normal sinus rhythm records
def get_label(codes):
    has_afib = "AFIB" in codes
    has_normal = "SR" in codes

    if has_afib and not has_normal:
        return 1
    if has_normal and not has_afib:
        return 0
    return None

df["label"] = df["scp_codes"].apply(get_label)
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

# Patient-level recommended PTB-XL splits
splits = {
    "train": (list(range(1, 9)), 400),
    "validation": ([9], 80),
    "test": ([10], 80)
}

selected = []

for split, (folds, n) in splits.items():
    for label in [0, 1]:
        subset = df[
            (df["strat_fold"].isin(folds)) &
            (df["label"] == label)
        ]

        sample = subset.sample(n=n, random_state=42).copy()
        sample["split"] = split
        selected.append(sample)

result = pd.concat(selected, ignore_index=True)

result.to_csv("data/processed/ecg_labels.csv", index=False)

print("Dataset prepared!")
print("\n", result.groupby(["split", "label"]).size())
print("\nTotal records:", len(result))

for code, count in counter.most_common():
    description = rhythm_codes.loc[code, "description"]
    print(f"{code:10} {description:35} {count}")

print("\nTotal records:", len(df))
print("Records with rhythm labels:", sum(
    1 for codes in df["scp_codes"]
    if any(code in rhythm_codes.index for code in codes)
))
