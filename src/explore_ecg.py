
import wfdb
import pandas as pd
import ast
import matplotlib.pyplot as plt

# Load one ECG
record = wfdb.rdrecord(
    "data/raw/ecg_100hz/records100/00000/00001_lr"
)

signals = record.p_signal
leads = record.sig_name
time = [i / record.fs for i in range(len(signals))]

# Plot all 12 leads
fig, axes = plt.subplots(12, 1, figsize=(14, 18), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(time, signals[:, i], linewidth=0.7)
    ax.set_ylabel(leads[i])
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("PTB-XL: 12-Lead ECG Visualization")
plt.tight_layout()
plt.savefig("results/sample_ecg.png", dpi=150)
plt.show()

# Inspect diagnostic labels
df = pd.read_csv("data/raw/ptbxl_database.csv")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

statements = pd.read_csv("data/raw/scp_statements.csv", index_col=0)

print("\nAvailable label categories:")
print(statements.columns.tolist())

print("\nDiagnostic classes:")
print(statements[statements["diagnostic"] == 1][
    ["description", "diagnostic_class"]
].head(20))

print("\nRhythm labels:")
print(statements[statements["rhythm"] == 1][
    ["description"]
].head(20))

print("\nFirst ECG labels:")
print(df.iloc[0]["scp_codes"])
