"""Plots for the folding-direction drafter sweep, in the team's figure style
(darkorange = static/prefix-blind, forestgreen = prefix-aware p=5; o- markers
with IQR error bars; red dashed enc-dec baseline at y=1).

Reads the per-protein CSVs written by folding_profile_sweep.py and renders a 2x2:
    [0,0] Mean k vs K            (== theoretical speedup; higher is faster)
    [0,1] Drafter accuracy vs #homologs
    [1,0] Mean k vs #homologs    (at K=5; shows the homolog sweet spot)
    [1,1] Per-draft acceptance vs K

Run:
    python3 folding_profile_plots.py --results folding_profile_sweep_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"static": "darkorange", "prefix": "forestgreen"}
LABELS = {"static": "static (prefix-blind)", "prefix": "prefix-aware (p=5)"}
N_SWEET = 32      # homolog count used for the K-axis panels
K_FIXED = 5       # K used for the N-axis speed panel
P_HEAD = 5        # prefix order shown as the headline prefix line


def _med_iqr(g):
    med = g.median()
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    return med, (med - q1), (q3 - med)


def _line(ax, x, med, lo, hi, mode, **kw):
    ax.errorbar(x, med.values, yerr=[lo.values, hi.values], fmt="o-",
                color=COLORS[mode], label=LABELS[mode], capsize=3, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="folding_profile_sweep_results")
    args = ap.parse_args()
    rdir = Path(args.results).resolve()

    k = pd.read_csv(rdir / "mean_k_sweep.csv")
    acc = pd.read_csv(rdir / "drafter_accuracy.csv")
    # normalise the prefix-order column (blank for static)
    for df in (k, acc):
        df["p"] = pd.to_numeric(df.get("p"), errors="coerce")

    def sel(df, mode):
        return df[df["mode"] == mode] if mode == "static" else df[(df["mode"] == "prefix") & (df["p"] == P_HEAD)]

    n_prot = k["protein_id"].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # [0,0] Mean k vs K  (== theoretical speedup), at N=N_SWEET
    ax = axes[0, 0]
    for mode in ("static", "prefix"):
        sub = sel(k[k["n_homologs"] == N_SWEET], mode)
        med, lo, hi = _med_iqr(sub.groupby("K")["mean_tokens_per_step"])
        _line(ax, med.index, med, lo, hi, mode)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.4, label="enc-dec (no spec.)")
    ax.set_xlabel("Draft length K")
    ax.set_ylabel("Mean k  (tokens / model call)")
    ax.set_title(f"Speedup vs K  —  N={N_SWEET} homologs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [0,1] Drafter accuracy vs #homologs
    ax = axes[0, 1]
    for mode in ("static", "prefix"):
        sub = sel(acc, mode)
        med, lo, hi = _med_iqr(sub.groupby("n_homologs")["drafter_accuracy"])
        _line(ax, med.index, med, lo, hi, mode)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("# homologs used")
    ax.set_ylabel("Drafter accuracy (vs ProstT5 greedy)")
    ax.set_title("Drafter accuracy vs #homologs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [1,0] Mean k vs #homologs at K=K_FIXED
    ax = axes[1, 0]
    for mode in ("static", "prefix"):
        sub = sel(k[k["K"] == K_FIXED], mode)
        med, lo, hi = _med_iqr(sub.groupby("n_homologs")["mean_tokens_per_step"])
        _line(ax, med.index, med, lo, hi, mode)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("# homologs used")
    ax.set_ylabel("Mean k  (tokens / model call)")
    ax.set_title(f"Speed vs #homologs  —  K={K_FIXED}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [1,1] Per-draft acceptance vs K, at N=N_SWEET
    ax = axes[1, 1]
    for mode in ("static", "prefix"):
        sub = sel(k[k["n_homologs"] == N_SWEET], mode)
        med, lo, hi = _med_iqr(sub.groupby("K")["acceptance_rate"])
        _line(ax, med.index, med, lo, hi, mode)
    ax.set_xlabel("Draft length K")
    ax.set_ylabel("Per-draft acceptance rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Acceptance vs K  —  N={N_SWEET} homologs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Folding (AA→3Di) profile drafter  —  {n_prot} proteins, "
                 f"prefix order p={P_HEAD}", fontsize=13)
    fig.tight_layout()
    outpath = rdir / "folding_profile_plots.png"
    fig.savefig(outpath, dpi=150)
    print(f"Wrote {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
