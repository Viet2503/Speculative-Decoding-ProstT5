"""Offline drafter-quality sweep for the AA->3Di FOLDING direction.

Goal (teammate's task): for the 40-protein `folding_MSA` dataset (64 homologs
each), measure DRAFTER ACCURACY and MEAN K of the folding HMM/profile drafter
across:
    * number of used homologs   N in {4, 8, 16, 32, 64}
    * fixed draft length        K in {1, 3, 5, 8, 11, 15}
    * prefix-context order      p in {static, 1, 2, 3, 4, 5}   ("p" knob)

Why this runs with NO GPU
-------------------------
Under *greedy* speculative decoding the generated sequence is provably identical
to plain enc-dec greedy, for any drafter. So:
    drafter accuracy = fraction of positions where the drafter's argmax 3Di token
                       equals ProstT5's greedy 3Di token at that position;
    mean k           = a deterministic simulation over that match pattern.
Both need only (a) the per-position drafter distribution and (b) ProstT5's greedy
3Di output per protein (the "reference"). The drafter distribution is rebuilt from
`homologs_projected_to_query_3di.fasta` — the dataset already folded every homolog
with ProstT5 and projected it to query columns, which is exactly what the team's
`FamilyFoldingHMMDrafter._build_logits` does internally. The reference is taken
from `query_3di.fasta` (which is itself ProstT5's greedy fold of the query).

This file replicates the team's drafter math from
`prostT5_probabilistic_drafter_folding.ipynb` (HMM-folding branch):
    HMM_SMOOTHING = 0.5, MAX_CONTEXT_P = 5, static counts + context_counts backoff,
    PrefixAwareFoldingHMMDrafter(max_p=p).

Run:
    python3 folding_profile_sweep.py --data ../folding_MSA \
        --out folding_profile_sweep_results
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- team constants (prostT5_probabilistic_drafter_folding.ipynb) -------------
HMM_SMOOTHING = 0.5
MAX_CONTEXT_P = 5
N_HOMOLOG_VALUES = [4, 8, 16, 32, 64]
K_VALUES = [1, 3, 5, 8, 11, 15]
P_VALUES = [1, 2, 3, 4, 5]            # prefix-context orders; "static" added too
GAP_CHARS = {"-", ".", "*"}


# --- IO -----------------------------------------------------------------------

def read_fasta(path: Path) -> list[tuple[str, str]]:
    recs: list[tuple[str, str]] = []
    name, buf = None, []
    for ln in path.read_text().splitlines():
        if ln.startswith(">"):
            if name is not None:
                recs.append((name, "".join(buf)))
            name, buf = ln[1:].split()[0] if ln[1:].split() else ln[1:], []
        else:
            buf.append(ln.strip())
    if name is not None:
        recs.append((name, "".join(buf)))
    return recs


def load_protein(d: Path):
    """Return (uid, query_3di_or_None, [projected_3di homolog strings])."""
    uid = d.name
    proj = [s for _, s in read_fasta(d / "homologs_projected_to_query_3di.fasta")]
    q3_path = d / "query_3di.fasta"
    q3 = read_fasta(q3_path)[0][1] if q3_path.exists() else None
    return uid, q3, proj


# --- drafter (faithful replica of FamilyFoldingHMMDrafter) --------------------

class FoldingProfileDrafter:
    """Static column counts + prefix context_counts, built from N projected
    homolog 3Di rows (already aligned to query columns)."""

    def __init__(self, projected_rows: list[str], L: int, sym_to_idx: dict[str, int],
                 smoothing: float = HMM_SMOOTHING, max_context_p: int = MAX_CONTEXT_P):
        self.L = L
        self.S = len(sym_to_idx)
        self.sym_to_idx = sym_to_idx
        self.smoothing = smoothing
        self.max_context_p = max_context_p

        counts = np.full((L, self.S), smoothing, dtype=np.float64)
        self.context_counts: dict[tuple[int, tuple], np.ndarray] = defaultdict(
            lambda: np.zeros(self.S, dtype=np.float64)
        )

        for row in projected_rows:
            projected = list(row[:L]) + [None] * max(0, L - len(row))
            projected = [c if c in sym_to_idx else None for c in projected]
            observed = [(pos, c) for pos, c in enumerate(projected) if c is not None]
            if not observed:
                continue
            for pos, tok in observed:
                counts[pos, sym_to_idx[tok]] += 1.0
            for pos, tok in observed:
                for ctx_len in range(min(max_context_p, pos), -1, -1):
                    ctx = projected[pos - ctx_len:pos]
                    if any(c is None for c in ctx):
                        continue
                    self.context_counts[(pos, tuple(ctx))][sym_to_idx[tok]] += 1.0

        self.static_counts = counts
        self.static_argmax = counts.argmax(axis=1)   # (L,)
        # drafter "confidence" = top-1 probability of the column distribution.
        # This is what HF's assistant_confidence_threshold compares against.
        self.static_conf = (counts.max(axis=1) / counts.sum(axis=1))  # (L,)

    def pred_conf_for_prefix(self, prefix_syms: list, max_p: int) -> tuple[int, float]:
        """(argmax class, top-1 prob) at position len(prefix), conditioned on up
        to max_p preceding reference symbols (with backoff). The probability is
        the drafter's confidence used by the dynamic-K policy."""
        pos = len(prefix_syms)
        if pos >= self.L:
            return -1, 0.0
        available = min(max_p, pos)
        for ctx_len in range(available, -1, -1):
            ctx = tuple(prefix_syms[-ctx_len:]) if ctx_len > 0 else tuple()
            key = (pos, ctx)
            if key in self.context_counts:
                c = self.context_counts[key] + self.smoothing
                return int(c.argmax()), float(c.max() / c.sum())
        return int(self.static_argmax[pos]), float(self.static_conf[pos])

    def argmax_for_prefix(self, prefix_syms: list, max_p: int) -> int:
        """argmax 3Di class at position len(prefix), conditioned on up to max_p
        preceding *reference* symbols (with backoff). Mirrors emission_for_prefix."""
        pos = len(prefix_syms)
        if pos >= self.L:
            return -1
        available = min(max_p, pos)
        for ctx_len in range(available, -1, -1):
            ctx = tuple(prefix_syms[-ctx_len:]) if ctx_len > 0 else tuple()
            key = (pos, ctx)
            if key in self.context_counts:
                return int(self.context_counts[key].argmax())
        return int(self.static_argmax[pos])


# --- greedy spec-decode simulation over a match mask --------------------------

def simulate_fixed_k(match: list[bool], K: int):
    """Greedy Leviathan accounting: each step proposes K drafts, accepts the
    leading run of matches, then emits one verifier token (+1). Returns
    (n_steps, mean_accepted, mean_tokens_per_step, acceptance_rate)."""
    L = len(match)
    cursor = steps = accepted_total = proposed_total = 0
    while cursor < L:
        k = min(K, L - cursor)
        a = 0
        while a < k and match[cursor + a]:
            a += 1
        accepted_total += a
        proposed_total += k
        steps += 1
        cursor += a + 1          # a accepted drafts + 1 verifier token (corr/bonus)
    mean_accepted = accepted_total / steps
    mean_tokens_per_step = L / steps          # tokens generated per model call == "mean k"
    acc_rate = accepted_total / proposed_total if proposed_total else 0.0
    return steps, mean_accepted, mean_tokens_per_step, acc_rate


# --- main sweep ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../folding_MSA",
                    help="path to the extracted folding_MSA/ directory")
    ap.add_argument("--out", default="folding_profile_sweep_results")
    ap.add_argument("--refs", default=None,
                    help="optional refs_3di.json (uid -> ProstT5 greedy 3Di). "
                         "Overrides query_3di and covers proteins missing it.")
    args = ap.parse_args()

    import json
    refs_json = json.loads(Path(args.refs).read_text()) if args.refs else {}

    data = Path(args.data).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dirs = sorted(p for p in data.iterdir() if p.is_dir())
    print(f"Loading {len(dirs)} proteins from {data}")

    # 3Di alphabet = sorted unique non-gap symbols across all projected rows
    alpha = set()
    loaded = []
    for d in dirs:
        uid, q3, proj = load_protein(d)
        loaded.append((uid, q3, proj))
        for row in proj:
            alpha |= {c for c in row if c not in GAP_CHARS}
    THREEDI = "".join(sorted(alpha))
    sym_to_idx = {c: i for i, c in enumerate(THREEDI)}
    print(f"3Di alphabet ({len(THREEDI)}): {THREEDI}")

    acc_rows = []     # per (protein, N, mode, p): drafter accuracy
    k_rows = []       # per (protein, N, mode, p, K): mean k etc.
    n_scored = 0
    n_noref = []

    for uid, q3, proj in loaded:
        ref_str = refs_json.get(uid) or q3
        ref_source = "refs_json" if (uid in refs_json) else "query_3di"
        if ref_str is None:
            n_noref.append(uid)
            continue
        L = len(ref_str)
        ref = list(ref_str)
        n_scored += 1

        for N in N_HOMOLOG_VALUES:
            rows_N = proj[:N]
            n_avail = len(rows_N)
            drafter = FoldingProfileDrafter(rows_N, L, sym_to_idx)

            # ----- static (prefix-blind) -----
            static_pred = drafter.static_argmax
            ref_idx = np.array([sym_to_idx.get(c, -1) for c in ref])
            match_static = [bool(static_pred[j] == ref_idx[j] and ref_idx[j] >= 0)
                            for j in range(L)]
            acc_static = float(np.mean(match_static))
            acc_rows.append(dict(protein_id=uid, length=L, n_homologs=N,
                                 n_homologs_avail=n_avail, mode="static", p="",
                                 drafter_accuracy=round(acc_static, 4),
                                 ref_source=ref_source))
            for K in K_VALUES:
                st, ma, mt, ar = simulate_fixed_k(match_static, K)
                k_rows.append(dict(protein_id=uid, length=L, n_homologs=N, mode="static",
                                   p="", K=K, drafter_accuracy=round(acc_static, 4),
                                   mean_accepted=round(ma, 4),
                                   mean_tokens_per_step=round(mt, 4),
                                   acceptance_rate=round(ar, 4), n_steps=st,
                                   ref_source=ref_source))

            # ----- prefix-aware p=1..5 -----
            for p in P_VALUES:
                match_p = []
                for j in range(L):
                    pred = drafter.argmax_for_prefix(ref[:j], p)
                    match_p.append(bool(pred == ref_idx[j] and ref_idx[j] >= 0))
                acc_p = float(np.mean(match_p))
                acc_rows.append(dict(protein_id=uid, length=L, n_homologs=N,
                                     n_homologs_avail=n_avail, mode="prefix", p=p,
                                     drafter_accuracy=round(acc_p, 4),
                                     ref_source=ref_source))
                for K in K_VALUES:
                    st, ma, mt, ar = simulate_fixed_k(match_p, K)
                    k_rows.append(dict(protein_id=uid, length=L, n_homologs=N,
                                       mode="prefix", p=p, K=K,
                                       drafter_accuracy=round(acc_p, 4),
                                       mean_accepted=round(ma, 4),
                                       mean_tokens_per_step=round(mt, 4),
                                       acceptance_rate=round(ar, 4), n_steps=st,
                                   ref_source=ref_source))

    # write per-protein CSVs
    _write_csv(out / "drafter_accuracy.csv", acc_rows)
    _write_csv(out / "mean_k_sweep.csv", k_rows)
    print(f"\nScored {n_scored} proteins (reference = query_3di greedy fold).")
    if n_noref:
        print(f"Skipped (no query_3di): {', '.join(n_noref)}")

    # ----- aggregate across proteins -----
    agg = _aggregate(k_rows)
    _write_csv(out / "summary_by_config.csv", agg)
    _print_headline(agg)

    print(f"\nWrote:\n  {out/'drafter_accuracy.csv'}\n  {out/'mean_k_sweep.csv'}"
          f"\n  {out/'summary_by_config.csv'}")
    return 0


def _aggregate(k_rows):
    groups = defaultdict(list)
    for r in k_rows:
        groups[(r["n_homologs"], r["mode"], r["p"], r["K"])].append(r)
    agg = []
    for (N, mode, p, K), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), str(kv[0][2]), kv[0][3])):
        agg.append(dict(
            n_homologs=N, mode=mode, p=p, K=K, n_proteins=len(rs),
            mean_drafter_accuracy=round(float(np.mean([r["drafter_accuracy"] for r in rs])), 4),
            mean_mean_k=round(float(np.mean([r["mean_tokens_per_step"] for r in rs])), 4),
            median_mean_k=round(float(np.median([r["mean_tokens_per_step"] for r in rs])), 4),
            mean_acceptance_rate=round(float(np.mean([r["acceptance_rate"] for r in rs])), 4),
        ))
    return agg


def _print_headline(agg):
    print("\n=== Drafter accuracy vs #homologs (static, K-independent) ===")
    seen = set()
    for r in agg:
        if r["mode"] == "static" and r["n_homologs"] not in seen:
            seen.add(r["n_homologs"])
            print(f"  N={r['n_homologs']:>2}: accuracy={r['mean_drafter_accuracy']:.3f}")
    print("\n=== Best mean-k config (mean tokens per model call, higher=faster) ===")
    best = max(agg, key=lambda r: r["mean_mean_k"])
    print(f"  N={best['n_homologs']} mode={best['mode']} p={best['p']} K={best['K']}"
          f"  -> mean_k={best['mean_mean_k']:.2f}  accuracy={best['mean_drafter_accuracy']:.3f}")
    print("\n=== mean_k at K=5, N=64, static vs prefix-p ===")
    for r in agg:
        if r["n_homologs"] == 64 and r["K"] == 5:
            tag = "static" if r["mode"] == "static" else f"p={r['p']}"
            print(f"  {tag:>8}: mean_k={r['mean_mean_k']:.2f}  acc={r['mean_drafter_accuracy']:.3f}")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
