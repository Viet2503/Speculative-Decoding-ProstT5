"""Dynamic-K policy comparison for the folding profile drafter (Gabriel's task:
"figure out the dynamic K algorithm").

Compares three draft-length policies, all simulated offline (greedy spec-decode
output is bit-identical to enc-dec, so this needs no GPU):

  1. fixed K                      — propose K every step (baseline).
  2. confidence threshold tau     — propose tokens while the drafter's top-1
                                    probability >= tau, capped at K_max. This is
                                    exactly HF's `assistant_confidence_threshold`,
                                    and it's what Jan asked for ("dynamic K
                                    dependent on the prediction confidence").
  3. acceptance heuristic         — HF's num_assistant_tokens_schedule="heuristic"
                                    (+2 on full accept, -1 otherwise). Confidence-
                                    blind; shown for contrast.

The point of (2): it only drafts tokens the drafter is confident about, so it
reaches a given mean k while proposing FEWER tokens than fixed K (higher draft
acceptance rate) — i.e. less wasted drafter work. That matters most for drafters
with per-token overhead (prefix-aware HMM, autoregressive CNN), which is exactly
where the team saw prefix-aware speedup < 1.

Run:
    python3 folding_dynamic_k.py --data prostT5_benchmarks/folding_MSA \
        --refs folding_profile_sweep_results/refs_3di.json \
        --out folding_profile_sweep_results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from folding_profile_sweep import FoldingProfileDrafter, GAP_CHARS, load_protein

N_HOMOLOGS = 32          # drafter sweet spot from the static sweep
K_MAX = 16               # cap on dynamic draft length (HF num_assistant_tokens)
FIXED_KS = [1, 2, 4, 8, 12, 16]
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CONFIGS = [("static", 0), ("prefix", 16)]   # (mode, prefix order p)
SCRIPT_DIR = Path(__file__).resolve().parent


# --- policy simulations (greedy Leviathan accounting) -------------------------

def sim_fixed(match, K):
    L = len(match); cursor = steps = prop = acc = ksum = 0
    while cursor < L:
        k = min(K, L - cursor)
        a = 0
        while a < k and match[cursor + a]:
            a += 1
        prop += k; acc += a; ksum += k; steps += 1; cursor += a + 1
    return steps, prop, acc, ksum


def sim_conf(match, conf, tau, kmax=K_MAX):
    L = len(match); cursor = steps = prop = acc = ksum = 0
    while cursor < L:
        k = 0
        while k < kmax and cursor + k < L and conf[cursor + k] >= tau:
            k += 1
        a = 0
        while a < k and match[cursor + a]:
            a += 1
        prop += k; acc += a; ksum += k; steps += 1; cursor += a + 1
    return steps, prop, acc, ksum


def sim_heuristic(match, k_init=5, k_min=1, k_max=K_MAX):
    L = len(match); cursor = steps = prop = acc = ksum = 0; kcur = k_init
    while cursor < L:
        k = min(kcur, L - cursor)
        a = 0
        while a < k and match[cursor + a]:
            a += 1
        prop += k; acc += a; ksum += k; steps += 1; cursor += a + 1
        kcur = min(kcur + 2, k_max) if a == k else max(kcur - 1, k_min)
    return steps, prop, acc, ksum


def per_protein_metrics(sim_out, L):
    steps, prop, acc, ksum = sim_out
    return dict(
        mean_k=L / steps,                              # tokens generated / model call
        proposed_per_step=prop / steps,                # drafter tokens proposed / step
        acceptance_rate=(acc / prop) if prop else 0.0, # accepted / proposed
        mean_K_used=ksum / steps,
    )


# --- build per-position match[] and conf[] for one (protein, config) ----------

def match_and_conf(drafter, ref_idx, ref_syms, mode, p):
    L = len(ref_syms)
    match, conf = [], []
    for j in range(L):
        if mode == "static":
            pred = int(drafter.static_argmax[j]); c = float(drafter.static_conf[j])
        else:
            pred, c = drafter.pred_conf_for_prefix(ref_syms[:j], p)
        match.append(bool(pred == ref_idx[j] and ref_idx[j] >= 0))
        conf.append(c)
    return match, conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(SCRIPT_DIR / "prostT5_benchmarks" / "folding_MSA"))
    ap.add_argument("--refs", default="folding_profile_sweep_results/refs_3di.json")
    ap.add_argument("--out", default="folding_profile_sweep_results")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    refs = json.loads(Path(args.refs).read_text()) if Path(args.refs).exists() else {}

    dirs = sorted(p for p in data.iterdir() if p.is_dir())
    loaded, alpha = [], set()
    for d in dirs:
        uid, q3, proj = load_protein(d)
        loaded.append((uid, q3, proj))
        for row in proj:
            alpha |= {c for c in row if c not in GAP_CHARS}
    sym_to_idx = {c: i for i, c in enumerate(sorted(alpha))}

    # accumulate per-protein metrics per (config, policy, param)
    bucket = {}   # key -> list of per-protein metric dicts
    def add(key, m):
        bucket.setdefault(key, []).append(m)

    n = 0
    for uid, q3, proj in loaded:
        ref_str = refs.get(uid) or q3
        if ref_str is None:
            continue
        n += 1
        L = len(ref_str)
        ref_syms = list(ref_str)
        ref_idx = np.array([sym_to_idx.get(c, -1) for c in ref_syms])
        rows_N = proj[:N_HOMOLOGS]
        if len(proj) < N_HOMOLOGS:
            print(f'protein "{uid}" has only {len(proj)} homologs, requires {N_HOMOLOGS}; using {len(rows_N)}.')
        drafter = FoldingProfileDrafter(rows_N, L, sym_to_idx)
        for mode, p in CONFIGS:
            match, conf = match_and_conf(drafter, ref_idx, ref_syms, mode, p)
            cfg = mode if mode == "static" else f"prefix_p{p}"
            for K in FIXED_KS:
                add((cfg, "fixed", K), per_protein_metrics(sim_fixed(match, K), L))
            for tau in TAUS:
                add((cfg, "conf", tau), per_protein_metrics(sim_conf(match, conf, tau), L))
            add((cfg, "heuristic", "k_init=5"),
                per_protein_metrics(sim_heuristic(match), L))

    # aggregate (mean across proteins)
    rows = []
    for (cfg, policy, param), ms in bucket.items():
        rows.append(dict(
            config=cfg, policy=policy, param=param, n_proteins=len(ms),
            mean_k=round(np.mean([m["mean_k"] for m in ms]), 4),
            proposed_per_step=round(np.mean([m["proposed_per_step"] for m in ms]), 4),
            acceptance_rate=round(np.mean([m["acceptance_rate"] for m in ms]), 4),
            mean_K_used=round(np.mean([m["mean_K_used"] for m in ms]), 4),
        ))
    rows.sort(key=lambda r: (r["config"], r["policy"], str(r["param"])))
    _write_csv(out / "dynamic_k_comparison.csv", rows)

    _plot(out / "dynamic_k_comparison.png", rows, n)
    _headline(rows, n)
    print(f"\nWrote {out/'dynamic_k_comparison.csv'} and {out/'dynamic_k_comparison.png'}")
    return 0


def _headline(rows, n):
    print(f"=== Dynamic-K comparison ({n} proteins, drafter: prefix p=16, N={N_HOMOLOGS}) ===")
    pr = [r for r in rows if r["config"] == "prefix_p16"]
    fixed = sorted([r for r in pr if r["policy"] == "fixed"], key=lambda r: r["param"])
    conf = sorted([r for r in pr if r["policy"] == "conf"], key=lambda r: r["param"])
    heur = [r for r in pr if r["policy"] == "heuristic"][0]
    print("\nfixed K:    " + "  ".join(f"K{r['param']}:k={r['mean_k']:.2f}/prop={r['proposed_per_step']:.1f}/acc={r['acceptance_rate']:.2f}" for r in fixed[:6]))
    print("conf tau:   " + "  ".join(f"t{r['param']}:k={r['mean_k']:.2f}/prop={r['proposed_per_step']:.1f}/acc={r['acceptance_rate']:.2f}" for r in conf))
    print(f"heuristic:  k={heur['mean_k']:.2f}/prop={heur['proposed_per_step']:.1f}/acc={heur['acceptance_rate']:.2f}/meanK={heur['mean_K_used']:.1f}")
    # at matched mean k ~ best fixed K=5, what does conf propose?
    target = next((r for r in fixed if r["param"] == 5), None)
    if target:
        near = min(conf, key=lambda r: abs(r["mean_k"] - target["mean_k"]))
        print(f"\nAt mean_k≈{target['mean_k']:.2f}: fixed K=5 proposes {target['proposed_per_step']:.1f}/step "
              f"(acc {target['acceptance_rate']:.2f}); conf tau={near['param']} reaches mean_k={near['mean_k']:.2f} "
              f"proposing {near['proposed_per_step']:.1f}/step (acc {near['acceptance_rate']:.2f}).")


def _plot(path, rows, n):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cfg, title in [(axes[0], "static", "static (prefix-blind)"),
                           (axes[1], "prefix_p16", "prefix-aware p=16")]:
        pr = [r for r in rows if r["config"] == cfg]
        fixed = sorted([r for r in pr if r["policy"] == "fixed"], key=lambda r: r["param"])
        conf = sorted([r for r in pr if r["policy"] == "conf"], key=lambda r: r["param"])
        heur = [r for r in pr if r["policy"] == "heuristic"][0]
        ax.plot([r["proposed_per_step"] for r in fixed], [r["mean_k"] for r in fixed],
                "o-", color="darkorange", label="fixed K")
        for r in fixed:
            ax.annotate(f"K{r['param']}", (r["proposed_per_step"], r["mean_k"]),
                        fontsize=7, color="darkorange", xytext=(2, -8), textcoords="offset points")
        ax.plot([r["proposed_per_step"] for r in conf], [r["mean_k"] for r in conf],
                "s-", color="forestgreen", label="confidence threshold τ")
        for r in conf:
            ax.annotate(f"{r['param']}", (r["proposed_per_step"], r["mean_k"]),
                        fontsize=7, color="forestgreen", xytext=(2, 4), textcoords="offset points")
        ax.plot([heur["proposed_per_step"]], [heur["mean_k"]], "*", color="crimson",
                markersize=15, label="acceptance heuristic")
        ax.set_xlabel("drafter tokens proposed / step  (∝ drafter overhead)")
        ax.set_ylabel("mean k  (tokens / model call)")
        ax.set_title(f"{title}  —  up-and-left is better")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Dynamic-K policy comparison — folding profile drafter, "
                 f"N={N_HOMOLOGS}, {n} proteins", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
