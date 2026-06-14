#!/usr/bin/env python3
"""
Plot FlexProfile speculative decoding benchmark results in CNN-style format.

Generates two output files:
- flexprofile_analysis_output1.png: Speedup, acceptance, wall time, and HF agreement (6 subplots)
- flexprofile_analysis_output2.png: α analysis and predicted tok/step ceiling (2 plots)

Usage:
    python plot_flexprofile_results.py              # All proteins (100)
    python plot_flexprofile_results.py --num-proteins 10  # First 10 proteins
    FLEXPROFILE_RUN=25_proteins python plot_flexprofile_results.py --num-proteins 5
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# Configuration
# Use 100_proteins by default, or specify via environment variable
RUN_NAME = os.environ.get("FLEXPROFILE_RUN", "100_proteins")
RESULTS_DIR = Path("prostT5/results/flexprofile") / RUN_NAME
CSV_FILE = RESULTS_DIR / "flexprofile_spec_decode_results.csv"
HF_CSV = RESULTS_DIR / "flexprofile_hf_assisted.csv"

BLUE, ORANGE, GRAY, GREEN = "#2e86de", "#f39c12", "#7f8c8d", "#27ae60"

def load_data():
    """Load benchmark results."""
    if not CSV_FILE.exists():
        print(f"Error: {CSV_FILE} not found")
        sys.exit(1)
    
    df = pd.read_csv(CSV_FILE)
    
    # Separate enc-dec and spec-decode results
    encdec_df = df[df["drafter"] == "enc_dec"].copy()
    spec_df = df[df["drafter"] == "flexprofile"].copy()
    
    # Load HF results if available
    hf_df = None
    if HF_CSV.exists():
        hf_df = pd.read_csv(HF_CSV)
    
    return df, encdec_df, spec_df, hf_df

def plot_output1(spec_df, hf_df, encdec_df, suffix=""):
    """Plot 6-subplot figure: speedup, acceptance, wall time, agreement.
    
    Args:
        suffix: String to append to output filename (e.g., "_10" for first 10 proteins)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Get K values
    K_VALUES = sorted(spec_df["K"].unique())
    
    # === Plot 1: Speedup Distribution by K (boxplot) ===
    ax1 = fig.add_subplot(gs[0, 0])
    data = [spec_df[spec_df["K"] == K]["speedup"].dropna().values for K in K_VALUES]
    bp = ax1.boxplot(data, patch_artist=True, widths=0.6)
    for patch, K in zip(bp["boxes"], K_VALUES):
        patch.set_facecolor(BLUE); patch.set_alpha(0.7)
    ax1.set_xticklabels([f"K={K}" for K in K_VALUES])
    ax1.axhline(1.0, color="red", ls="--", alpha=0.5, label="no speedup (1×)")
    ax1.set_ylabel("Speedup vs enc-dec")
    ax1.set_title("Speedup Distribution by Draft Length K")
    ax1.legend(fontsize=8)
    
    # === Plot 2: Acceptance Rate Distribution (histogram) ===
    ax2 = fig.add_subplot(gs[0, 1])
    alpha_vals = spec_df[spec_df["K"] == 1]["draft_acceptance_rate"].dropna()
    ax2.hist(alpha_vals, bins=20, color=BLUE, alpha=0.7, edgecolor="black")
    ax2.axvline(alpha_vals.mean(), color="red", linestyle="--", linewidth=2, label=f"mean α = {alpha_vals.mean():.3f}")
    ax2.set_xlabel("Per-protein acceptance rate α")
    ax2.set_ylabel("Number of proteins")
    ax2.set_title("Distribution of Acceptance Rate α (Greedy)")
    ax2.legend(fontsize=9)
    
    # === Plot 3: Predicted vs Measured Tok/Step ===
    ax3 = fig.add_subplot(gs[0, 2])
    alpha_mean = alpha_vals.mean()
    pred_tps = [(1 - alpha_mean**(K + 1)) / (1 - alpha_mean) if alpha_mean < 0.999 else K + 1 for K in K_VALUES]
    meas_tps = [spec_df[spec_df["K"] == K]["mean_tps"].mean() for K in K_VALUES]
    
    x = np.arange(len(K_VALUES))
    width = 0.35
    ax3.bar(x - width/2, pred_tps, width, label="Pred tok/step (Thm 3.8)", color=BLUE, alpha=0.8)
    ax3.bar(x + width/2, meas_tps, width, label="Measured speedup", color=ORANGE, alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"K={K}" for K in K_VALUES])
    ax3.set_ylabel("Tokens/step")
    ax3.set_title("Predicted Tok/Step vs Measured Speedup")
    ax3.legend(fontsize=9)
    
    # === Plot 4: Wall Time vs Protein Length ===
    ax4 = fig.add_subplot(gs[1, 0])
    # Merge to get lengths and times
    merge_df = spec_df.merge(encdec_df[["protein_id", "wall_s"]], on="protein_id", how="left", suffixes=("_spec", "_enc"))
    for K in K_VALUES:
        sub = merge_df[merge_df["K"] == K]
        ax4.scatter(sub["length"], sub["wall_s_spec"], label=f"HF K={K}", alpha=0.6, s=30)
    enc_only = encdec_df[["length", "wall_s"]].drop_duplicates(subset=["length"])
    ax4.scatter(enc_only["length"], enc_only["wall_s"], label="enc-dec (baseline)", alpha=0.5, s=30, color=GRAY)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Protein length (AA residues)")
    ax4.set_ylabel("Wall time (s)")
    ax4.set_title("Wall Time vs Protein Length")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # === Plot 5: Speedup vs Length colored by α ===
    ax5 = fig.add_subplot(gs[1, 1])
    K_med = 4
    sub = spec_df[spec_df["K"] == K_med].copy()
    scatter = ax5.scatter(sub["length"], sub["speedup"], c=sub["draft_acceptance_rate"], 
                         cmap="RdYlGn", s=50, alpha=0.7, vmin=0, vmax=1)
    ax5.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Protein length (AA residues)")
    ax5.set_ylabel(f"Speedup (K={K_med})")
    ax5.set_title(f"Speedup vs Length (K={K_med}, coloured by α)")
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label("α")
    ax5.grid(True, alpha=0.3)
    
    # === Plot 6: HF Agreement ===
    ax6 = fig.add_subplot(gs[1, 2])
    if hf_df is not None and len(hf_df) > 0:
        # HF results don't have K values, so compute overall agreement
        hf_ok_count = hf_df["hf_ok"].sum()
        hf_matches = hf_df[hf_df["hf_ok"]]["hf_matches_ref"].sum() if hf_ok_count > 0 else 0
        match_pct = (hf_matches / hf_ok_count * 100) if hf_ok_count > 0 else 0
        
        bars = ax6.bar(["Overall"], [match_pct], color=BLUE, alpha=0.8, width=0.4)
        ax6.axhline(100, color="green", linestyle="--", alpha=0.5, label="expected (100%)")
        ax6.set_ylim(0, 110)
        ax6.set_ylabel("Output matches plain greedy (%)")
        ax6.set_title("HF Output Agreement with enc-dec Greedy")
        ax6.legend(fontsize=8)
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2, height + 2,
                    f"{height:.1f}%", ha="center", va="bottom", fontsize=10)
    
    plt.suptitle("ProstT5 Speculative Decoding — Folding Direction (AA → 3Di)\nFlexProfile Drafter via Custom Implementation",
                 fontsize=13, y=0.995)
    
    out_path = RESULTS_DIR / f"flexprofile_analysis_output1{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

def plot_output2(spec_df, suffix=""):
    """Plot α analysis: α vs length, and predicted tok/step ceiling.
    
    Args:
        suffix: String to append to output filename (e.g., "_10" for first 10 proteins)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    K_VALUES = sorted(spec_df["K"].unique())
    
    # === Plot 1: α vs Protein Length ===
    ax1 = axes[0]
    alpha_by_protein = spec_df[spec_df["K"] == 1][["protein_id", "length", "draft_acceptance_rate"]].drop_duplicates()
    ax1.scatter(alpha_by_protein["length"], alpha_by_protein["draft_acceptance_rate"], 
               s=40, alpha=0.6, color=BLUE)
    alpha_mean = alpha_by_protein["draft_acceptance_rate"].mean()
    ax1.axhline(alpha_mean, color="red", linestyle="--", linewidth=2, label=f"mean α = {alpha_mean:.3f}")
    ax1.set_xlabel("Protein length (AA residues)")
    ax1.set_ylabel("Acceptance rate α")
    ax1.set_title("α vs Protein Length")
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Predicted Tok/Step Ceiling by K and α ===
    ax2 = axes[1]
    K_range = np.arange(1, 17)
    alpha_vals = [0.10, 0.20, 0.30, 0.40, 0.50, alpha_mean]
    colors_alpha = plt.cm.viridis(np.linspace(0, 1, len(alpha_vals)))
    
    for alpha, color in zip(alpha_vals, colors_alpha):
        if alpha == alpha_mean:
            pred = [(1 - alpha**(K + 1)) / (1 - alpha) if alpha < 0.999 else K + 1 for K in K_range]
            ax2.plot(K_range, pred, "k--", linewidth=3, label=f"Actual mean α = {alpha:.3f}", marker="o")
        else:
            pred = [(1 - alpha**(K + 1)) / (1 - alpha) if alpha < 0.999 else K + 1 for K in K_range]
            ax2.plot(K_range, pred, "--", color=color, alpha=0.6, label=f"α = {alpha:.2f}")
    
    ax2.set_xlabel("Draft length K")
    ax2.set_ylabel("Expected tokens/step")
    ax2.set_title("Predicted Tok/Step Ceiling by K and α")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("α Analysis — Folding Direction (AA → 3Di)", fontsize=12, y=1.00)
    
    out_path = RESULTS_DIR / f"flexprofile_analysis_output2{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot FlexProfile speculative decoding benchmark results"
    )
    parser.add_argument(
        "--num-proteins", "-n",
        type=int,
        default=None,
        help="Only plot first N proteins (default: all)"
    )
    args = parser.parse_args()
    
    print(f"Loading results from {RESULTS_DIR}...")
    df, encdec_df, spec_df, hf_df = load_data()
    
    # Get unique protein IDs in order they appear in data
    protein_ids = spec_df["protein_id"].unique()
    
    # Filter to first N proteins if specified
    suffix = ""
    if args.num_proteins is not None:
        protein_ids = protein_ids[:args.num_proteins]
        suffix = f"_{args.num_proteins}"
        spec_df = spec_df[spec_df["protein_id"].isin(protein_ids)]
        encdec_df = encdec_df[encdec_df["protein_id"].isin(protein_ids)]
        if hf_df is not None:
            hf_df = hf_df[hf_df["id"].isin(protein_ids)]
    
    print(f"Benchmark results:")
    print(f"  Total records: {len(spec_df) + len(encdec_df)}")
    print(f"  Proteins: {spec_df['protein_id'].nunique()}")
    print(f"  K values: {sorted(spec_df['K'].unique())}")
    print(f"  Proteins with HF results: {hf_df['id'].nunique() if hf_df is not None else 0}")
    
    print("\nGenerating plots...")
    plot_output1(spec_df, hf_df, encdec_df, suffix=suffix)
    plot_output2(spec_df, suffix=suffix)
    print("Done!")
