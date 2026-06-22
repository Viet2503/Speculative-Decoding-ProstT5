"""
Aggregate FlexProfile results.

Prints a short summary table for each draft length `K` found in
`flexprofile_spec_decode_results.csv`.
"""

import os
from pathlib import Path

import pandas as pd

# @title Aggregate results

RUN_NAME = os.environ.get("FLEXPROFILE_RUN", "100_proteins")
# RUN_NAME = "first_20_proteins"
RESULTS_DIR = Path("prostT5/results/flexprofile") / RUN_NAME
CSV_FILE = RESULTS_DIR / "flexprofile_spec_decode_results.csv"
df = pd.read_csv(CSV_FILE)

encdec_df = df[df["drafter"] == "enc_dec"]
spec_df = df[df["drafter"] == "flexprofile"]

K_VALUES = sorted(spec_df["K"].unique().tolist())

print("=== Speedup by K (flexprofile drafter) ===")
print(f"{'K':>7s}  {'Mean Speedup':>12s}  {'Median':>8s}  {'Draft accept':>12s}  {'Tok/step':>8s}")
print("-" * 55)
for K in K_VALUES:
    sub = spec_df[spec_df["K"] == K]
    if len(sub):
        print(f"{K:7d}  {sub['speedup'].mean():12.2f}  {sub['speedup'].median():8.2f}  "
              f"{sub['draft_acceptance_rate'].mean():12.3f}  {sub['mean_tps'].mean():8.2f}")
