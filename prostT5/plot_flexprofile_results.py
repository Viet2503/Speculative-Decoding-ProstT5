#!/usr/bin/env python3
"""
Plot FlexProfile speculative decoding benchmark results (CNN folding style).

Generates three PNG files in the results directory:
- flexprofile_alpha_analysis.png      — α vs length + predicted tok/step ceiling
- flexprofile_spec_dec_plots.png        — 6-panel speedup / acceptance / wall-time dashboard
- flexprofile_per_protein_speedup.png   — per-protein speedup for all K values

Usage:
    python plot_flexprofile_results.py
    python plot_flexprofile_results.py --num-proteins 20
    FLEXPROFILE_RUN=first_20_proteins python plot_flexprofile_results.py
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_NAME = os.environ.get("FLEXPROFILE_RUN", "100_proteins")
# RUN_NAME = "first_20_proteins"
RESULTS_DIR = Path("prostT5/results/flexprofile") / RUN_NAME
CSV_FILE = RESULTS_DIR / "flexprofile_spec_decode_results.csv"
HF_CSV = RESULTS_DIR / "flexprofile_hf_assisted.csv"
ONESHOT_CSV = RESULTS_DIR / "flexprofile_oneshot.csv"

BLUE, ORANGE, RED, GRAY, GREEN = "#2e86de", "#f39c12", "#e74c3c", "#7f8c8d", "#27ae60"
K_COLORS = {1: BLUE, 2: ORANGE, 4: RED, 8: GRAY}


def pred_tok_per_step(alpha: float, k: int) -> float:
    if alpha >= 0.999:
        return float(k + 1)
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def load_data():
    if not CSV_FILE.exists():
        print(f"Error: {CSV_FILE} not found")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE)
    encdec_df = df[df["drafter"] == "enc_dec"].copy()
    spec_df = df[df["drafter"] == "flexprofile"].copy()

    hf_df = pd.read_csv(HF_CSV) if HF_CSV.exists() else None
    oneshot_df = pd.read_csv(ONESHOT_CSV) if ONESHOT_CSV.exists() else None
    return df, encdec_df, spec_df, hf_df, oneshot_df


def alpha_per_protein(spec_df: pd.DataFrame) -> pd.DataFrame:
    """One greedy α per protein from custom spec-decode at K=1.

    K=1 draft_acceptance_rate ≈ static enc-dec↔flex argmax agreement (Leviathan α).
    Do NOT use flexprofile_hf_assisted.csv here: its draft_acceptance_rate is measured
    at HF_K (4 in the benchmark notebook), so it is much lower than true α.
    """
    out = (
        spec_df[spec_df["K"] == 1][["protein_id", "length", "draft_acceptance_rate"]]
        .drop_duplicates(subset=["protein_id"])
    )
    if "draft_acceptance_std" in spec_df.columns:
        std_col = spec_df[spec_df["K"] == 1][["protein_id", "draft_acceptance_std"]].drop_duplicates(
            subset=["protein_id"]
        )
        out = out.merge(std_col, on="protein_id", how="left")
    return out


def plot_alpha_analysis(spec_df: pd.DataFrame, suffix: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    alpha_df = alpha_per_protein(spec_df)
    alpha_mean = alpha_df["draft_acceptance_rate"].mean()
    alpha_std = alpha_df["draft_acceptance_rate"].std()

    ax1 = axes[0]
    yerr = None
    subtitle = "(K=1 custom spec-decode, per-protein greedy α)"
    if "draft_acceptance_std" in alpha_df.columns and alpha_df["draft_acceptance_std"].notna().any():
        yerr = alpha_df["draft_acceptance_std"]
        subtitle = "(error bars = std over residues within protein)"
    ax1.errorbar(
        alpha_df["length"],
        alpha_df["draft_acceptance_rate"],
        yerr=yerr,
        fmt="o",
        color=BLUE,
        alpha=0.65,
        markersize=5,
        capsize=2,
        linestyle="none",
    )
    ax1.axhline(alpha_mean, color="red", linestyle="--", linewidth=2, label=f"mean α = {alpha_mean:.3f}")
    ax1.set_xlabel("Protein length (AA residues)")
    ax1.set_ylabel("Acceptance rate α")
    ax1.set_title(f"α vs Protein Length\n{subtitle}")
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    k_range = np.arange(1, 17)
    ref_alphas = np.quantile(alpha_df["draft_acceptance_rate"], [0.1, 0.3, 0.5, 0.7, 0.9])
    ref_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(ref_alphas)))

    for alpha_ref, color in zip(ref_alphas, ref_colors):
        pred = [pred_tok_per_step(alpha_ref, int(k)) for k in k_range]
        ax2.plot(k_range, pred, "--", color=color, alpha=0.75, linewidth=1.5, label=f"α = {alpha_ref:.2f}")

    pred_mean = [pred_tok_per_step(alpha_mean, int(k)) for k in k_range]
    ax2.plot(
        k_range,
        pred_mean,
        "k--",
        linewidth=3,
        marker="o",
        markersize=4,
        label=f"Actual mean α = {alpha_mean:.3f}",
    )

    if np.isfinite(alpha_std) and alpha_std > 0:
        alpha_lo = max(0.0, alpha_mean - alpha_std)
        alpha_hi = min(0.999, alpha_mean + alpha_std)
        pred_lo = [pred_tok_per_step(alpha_lo, int(k)) for k in k_range]
        pred_hi = [pred_tok_per_step(alpha_hi, int(k)) for k in k_range]
        ax2.fill_between(k_range, pred_lo, pred_hi, color="gray", alpha=0.25, label=f"α ∈ [{alpha_lo:.2f}, {alpha_hi:.2f}]")

    for k_mark in (2, 4, 8):
        if k_mark <= k_range[-1]:
            ax2.axvline(k_mark, color="gray", linestyle=":", alpha=0.4)

    ax2.set_xlabel("Draft length K")
    ax2.set_ylabel("Expected tokens/step")
    ax2.set_title("Predicted Tok/Step Ceiling by K and α\n(shaded band = ± 1 std of observed α)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("α Analysis — Folding Direction (AA → 3Di)", fontsize=12, y=1.02)
    out_path = RESULTS_DIR / f"flexprofile_alpha_analysis{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_spec_dec_plots(
    spec_df: pd.DataFrame,
    encdec_df: pd.DataFrame,
    oneshot_df: pd.DataFrame | None,
    suffix: str = "",
):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.32)
    k_values = sorted(spec_df["K"].unique())
    alpha_df = alpha_per_protein(spec_df)
    alpha_mean = alpha_df["draft_acceptance_rate"].mean()
    alpha_std = alpha_df["draft_acceptance_rate"].std()

    ax1 = fig.add_subplot(gs[0, 0])
    speedup_data = [spec_df[spec_df["K"] == k]["speedup"].dropna().values for k in k_values]
    bp = ax1.boxplot(
        speedup_data,
        patch_artist=True,
        widths=0.6,
        whis=[5, 95],
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
    )
    for patch, k in zip(bp["boxes"], k_values):
        patch.set_facecolor(K_COLORS.get(k, GRAY))
        patch.set_alpha(0.75)
    ax1.set_xticklabels([f"K={k}" for k in k_values])
    ax1.axhline(1.0, color="red", ls="--", alpha=0.5, label="no speedup (1×)")
    ax1.set_ylabel("Speedup over plain enc-dec")
    ax1.set_title("Speedup Distribution by K")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    alpha_vals = alpha_df["draft_acceptance_rate"].dropna()
    ax2.hist(alpha_vals, bins=20, color=BLUE, alpha=0.7, edgecolor="black")
    ax2.axvline(alpha_mean, color="red", linestyle="--", linewidth=2, label=f"mean α = {alpha_mean:.3f}")
    if np.isfinite(alpha_std) and alpha_std > 0:
        ax2.axvspan(alpha_mean - alpha_std, alpha_mean + alpha_std, color="red", alpha=0.15, label=f"± 1 std ({alpha_std:.3f})")
    ax2.set_xlabel("Per-protein acceptance rate α")
    ax2.set_ylabel("Number of proteins")
    ax2.set_title("Distribution of Acceptance Rate α")
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    pred_tps = [pred_tok_per_step(alpha_mean, int(k)) for k in k_values]
    meas_mean = [spec_df[spec_df["K"] == k]["speedup"].mean() for k in k_values]
    meas_std = [spec_df[spec_df["K"] == k]["speedup"].std() for k in k_values]
    x = np.arange(len(k_values))
    width = 0.35
    ax3.bar(x - width / 2, pred_tps, width, label="Pred tok/step (Thm 3.8)", color=BLUE, alpha=0.85)
    ax3.bar(
        x + width / 2,
        meas_mean,
        width,
        yerr=meas_std,
        capsize=4,
        label="Measured speedup (mean ± std)",
        color=ORANGE,
        alpha=0.85,
        error_kw={"ecolor": "black", "elinewidth": 1},
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"K={k}" for k in k_values])
    ax3.set_ylabel("Tokens/step | Speedup")
    ax3.set_title("Predicted Tok/Step vs Measured Speedup")
    ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(gs[1, 0])
    merge_df = spec_df.merge(
        encdec_df[["protein_id", "wall_s"]],
        on="protein_id",
        how="left",
        suffixes=("_spec", "_enc"),
    )
    for k in k_values:
        sub = merge_df[merge_df["K"] == k]
        ax4.scatter(
            sub["length"],
            sub["wall_s_spec"],
            label=f"HF K={k}",
            alpha=0.6,
            s=30,
            color=K_COLORS.get(k, GRAY),
        )
    enc_only = encdec_df[["length", "wall_s"]].drop_duplicates(subset=["length"])
    ax4.scatter(enc_only["length"], enc_only["wall_s"], label="enc-dec (baseline)", alpha=0.5, s=30, color=GRAY)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("Protein length (AA residues)")
    ax4.set_ylabel("Wall time (s)")
    ax4.set_title("Wall Time vs Protein Length")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    k_highlight = max(k_values)
    sub = spec_df[spec_df["K"] == k_highlight].copy()
    scatter = ax5.scatter(
        sub["length"],
        sub["speedup"],
        c=sub["draft_acceptance_rate"],
        cmap="RdYlGn",
        s=50,
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    ax5.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Protein length (AA residues)")
    ax5.set_ylabel(f"Speedup (K={k_highlight})")
    ax5.set_title(f"Speedup vs Length (K={k_highlight}, coloured by α)")
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label("α")
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    if oneshot_df is not None and "flex_vs_afdb_3di" in oneshot_df.columns:
        recovery = oneshot_df[["id", "length", "flex_vs_afdb_3di"]].rename(columns={"id": "protein_id"})
        for k in k_values:
            jitter = (k_values.index(k) - len(k_values) / 2) * 3
            ax6.scatter(
                recovery["length"] + jitter,
                recovery["flex_vs_afdb_3di"],
                alpha=0.45,
                s=25,
                color=K_COLORS.get(k, GRAY),
                label=f"K={k}",
            )
        ax6.set_xlabel("Protein length (AA residues)")
        ax6.set_ylabel("Sequence recovery (vs ground-truth 3Di)")
        ax6.set_title("Sequence Recovery vs Length")
        ax6.set_ylim(0, 1.05)
        ax6.legend(fontsize=8, ncol=2)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "flexprofile_oneshot.csv\nnot found", ha="center", va="center", transform=ax6.transAxes)

    plt.suptitle(
        "ProstT5 Speculative Decoding — Folding Direction (AA → 3Di)\n"
        "FlexProfile Drafter via Custom Implementation",
        fontsize=13,
        y=0.995,
    )
    out_path = RESULTS_DIR / f"flexprofile_spec_dec_plots{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_per_protein_speedup(spec_df: pd.DataFrame, suffix: str = ""):
    fig, ax = plt.subplots(figsize=(10, 6))
    k_values = sorted(spec_df["K"].unique())

    ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="no speedup (1×)")
    for k in k_values:
        sub = spec_df[spec_df["K"] == k]
        ax.scatter(
            sub["length"],
            sub["speedup"],
            label=f"K={k}",
            alpha=0.55,
            s=40,
            color=K_COLORS.get(k, GRAY),
        )

    ax.set_xlabel("Protein length (AA residues)")
    ax.set_ylabel("Speedup over plain enc-dec")
    ax.set_title("Per-Protein Speedup — All K Values (Folding Direction)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = RESULTS_DIR / f"flexprofile_per_protein_speedup{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot FlexProfile speculative decoding benchmark results")
    parser.add_argument("--num-proteins", "-n", type=int, default=None, help="Only plot first N proteins")
    args = parser.parse_args()

    print(f"Loading results from {RESULTS_DIR}...")
    _, encdec_df, spec_df, hf_df, oneshot_df = load_data()

    suffix = ""
    if args.num_proteins is not None:
        protein_ids = spec_df["protein_id"].unique()[: args.num_proteins]
        suffix = f"_{args.num_proteins}"
        spec_df = spec_df[spec_df["protein_id"].isin(protein_ids)]
        encdec_df = encdec_df[encdec_df["protein_id"].isin(protein_ids)]
        if hf_df is not None:
            hf_df = hf_df[hf_df["id"].isin(protein_ids)]
        if oneshot_df is not None:
            oneshot_df = oneshot_df[oneshot_df["id"].isin(protein_ids)]

    print("Benchmark results:")
    print(f"  Proteins: {spec_df['protein_id'].nunique()}")
    print(f"  K values: {sorted(spec_df['K'].unique())}")
    print(f"  HF rows: {len(hf_df) if hf_df is not None else 0}")

    print("\nGenerating plots...")
    plot_alpha_analysis(spec_df, suffix=suffix)
    plot_spec_dec_plots(spec_df, encdec_df, oneshot_df, suffix=suffix)
    plot_per_protein_speedup(spec_df, suffix=suffix)
    print("Done!")


if __name__ == "__main__":
    main()
