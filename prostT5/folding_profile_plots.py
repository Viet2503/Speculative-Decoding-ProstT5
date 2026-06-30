"""Plots for the folding-direction drafter sweep, in the team's figure style
(darkorange = static/prefix-blind, forestgreen = prefix-aware p=16; o- markers
with IQR error bars; red dashed enc-dec baseline at y=1).

Reads the per-protein CSVs written by folding_profile_sweep.py and renders a 2x2:
    [0,0] Mean k vs K            (== theoretical speedup; higher is faster)
    [0,1] Drafter accuracy vs #homologs
    [1,0] Mean k vs #homologs    (at K=4; shows the homolog sweet spot)
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
N_SWEET = 32      # homolog count used for the K-axis panels
K_FIXED = 4       # K used for the N-axis speed panel
P_HEAD = 16       # prefix order shown as the headline prefix line
LABELS = {"static": "static (prefix-blind)", "prefix": f"prefix-aware (p={P_HEAD})"}


def _med_iqr(g):
    med = g.median()
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    return med, (med - q1), (q3 - med)


def _line(ax, x, med, lo, hi, mode, **kw):
    ax.errorbar(x, med.values, yerr=[lo.values, hi.values], fmt="o-",
                color=COLORS[mode], label=LABELS[mode], capsize=3, **kw)


def grid_figure(rdir, summ, n_prot) -> None:
    """A 2x3 figure exposing all three knobs (prefix order p, draft length K,
    #homologs) for both metrics. Uses the across-protein means in
    summary_by_config.csv. orange=static; green gradient=p1..5; plasma=#homologs."""
    import numpy as np

    N_VALUES = sorted(int(x) for x in summ["n_homologs"].unique())
    K_VALUES = sorted(int(x) for x in summ["K"].unique())
    K0 = K_VALUES[0]                      # accuracy is K-independent; use any K
    P_ORDER = ["static"] + sorted(int(x) for x in summ.loc[summ["mode"] == "prefix", "p"].dropna().unique())
    green = plt.cm.Greens

    p_values = [p for p in P_ORDER if p != "static"]
    p_denom = max(len(p_values) - 1, 1)

    def p_color(c): return "darkorange" if c == "static" else green(0.35 + 0.6 * p_values.index(c) / p_denom)
    def p_fmt(c):   return "o--" if c == "static" else "o-"
    def p_label(c): return "static" if c == "static" else f"p={c}"
    n_denom = max(len(N_VALUES) - 1, 1)

    def n_color(n): return plt.cm.plasma(0.08 + 0.74 * N_VALUES.index(n) / n_denom)

    def cell(metric, cfg, by, **fixed):
        s = summ[summ["mode"] == "static"] if cfg == "static" \
            else summ[(summ["mode"] == "prefix") & (summ["p"] == cfg)]
        for col, val in fixed.items():
            s = s[s[col] == val]
        return s.set_index(by)[metric]

    def logx(ax):
        ax.set_xscale("log", base=2); ax.set_xticks(N_VALUES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # [0,0] accuracy vs #homologs, one line per p
    ax = axes[0, 0]
    for c in P_ORDER:
        s = cell("mean_drafter_accuracy", c, "n_homologs", K=K0)
        ax.plot(N_VALUES, [s.get(n) for n in N_VALUES], p_fmt(c), color=p_color(c), label=p_label(c))
    logx(ax); ax.set_xlabel("# homologs"); ax.set_ylabel("Drafter accuracy")
    ax.set_title("Accuracy vs #homologs  (per p)"); ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    # [0,1] accuracy vs p, one line per #homologs
    ax = axes[0, 1]
    xs = list(range(len(P_ORDER)))
    for n in N_VALUES:
        ys = [cell("mean_drafter_accuracy", c, "n_homologs", K=K0).get(n) for c in P_ORDER]
        ax.plot(xs, ys, "o-", color=n_color(n), label=f"N={n}")
    ax.set_xticks(xs); ax.set_xticklabels([p_label(c) for c in P_ORDER])
    ax.set_xlabel("prefix order p"); ax.set_ylabel("Drafter accuracy")
    ax.set_title("Accuracy vs p  (per #homologs)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [0,2] accuracy heatmap p x #homologs
    ax = axes[0, 2]
    M = np.array([[cell("mean_drafter_accuracy", c, "n_homologs", K=K0).get(n) for n in N_VALUES]
                  for c in P_ORDER], dtype=float)
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(N_VALUES))); ax.set_xticklabels(N_VALUES)
    ax.set_yticks(range(len(P_ORDER))); ax.set_yticklabels([p_label(c) for c in P_ORDER])
    ax.set_xlabel("# homologs"); ax.set_ylabel("prefix order p"); ax.set_title("Accuracy heatmap")
    thr = (M.max() + M.min()) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # [1,0] mean k vs K, one line per p  (N=32)
    ax = axes[1, 0]
    for c in P_ORDER:
        s = cell("mean_mean_k", c, "K", n_homologs=32)
        ax.plot(K_VALUES, [s.get(k) for k in K_VALUES], p_fmt(c), color=p_color(c), label=p_label(c))
    ax.axhline(1.0, color="red", ls="--", alpha=0.4)
    ax.set_xlabel("Draft length K"); ax.set_ylabel("Mean k  (tokens / model call)")
    ax.set_title("Mean k vs K  (per p, N=32)"); ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    # [1,1] mean k vs K, one line per #homologs  (p=P_HEAD)
    ax = axes[1, 1]
    for n in N_VALUES:
        s = cell("mean_mean_k", P_HEAD, "K", n_homologs=n)
        ax.plot(K_VALUES, [s.get(k) for k in K_VALUES], "o-", color=n_color(n), label=f"N={n}")
    ax.axhline(1.0, color="red", ls="--", alpha=0.4)
    ax.set_xlabel("Draft length K"); ax.set_ylabel("Mean k  (tokens / model call)")
    ax.set_title(f"Mean k vs K  (per #homologs, p={P_HEAD})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [1,2] mean k vs #homologs, one line per p  (K=5)
    ax = axes[1, 2]
    for c in P_ORDER:
        s = cell("mean_mean_k", c, "n_homologs", K=5)
        ax.plot(N_VALUES, [s.get(n) for n in N_VALUES], p_fmt(c), color=p_color(c), label=p_label(c))
    logx(ax); ax.set_xlabel("# homologs"); ax.set_ylabel("Mean k  (tokens / model call)")
    ax.set_title("Mean k vs #homologs  (per p, K=5)"); ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Folding (AA→3Di) drafter sweep — all knobs (p × K × #homologs) — "
                 f"{n_prot} proteins", fontsize=14)
    fig.tight_layout()
    out = rdir / "folding_profile_grid.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


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
    ax.set_xticks(sorted(int(x) for x in acc["n_homologs"].unique()))
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
    ax.set_xticks(sorted(int(x) for x in k["n_homologs"].unique()))
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

    # full p × K × #homologs grid
    summ = pd.read_csv(rdir / "summary_by_config.csv")
    summ["p"] = pd.to_numeric(summ["p"], errors="coerce")
    grid_figure(rdir, summ, n_prot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
