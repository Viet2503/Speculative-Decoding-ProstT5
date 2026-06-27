# Folding-direction drafter sweep — results

Task (from teammate): on the 40-protein `folding_MSA` dataset (64 homologs each,
AA→3Di **folding** direction), measure **drafter accuracy** and **mean k** of the
folding profile/HMM drafter across number-of-homologs, K, and prefix-context `p`.

Produced by `prostT5/folding_profile_sweep.py` — a GPU-free offline replica of the
team's `FamilyFoldingHMMDrafter` (`prostT5_probabilistic_drafter_folding.ipynb`,
HMM-folding branch). Drafter counts come straight from
`homologs_projected_to_query_3di.fasta` (the dataset already folded every homolog
with ProstT5 and projected it to query columns). Reference = `query_3di.fasta`.

- **drafter accuracy** = fraction of positions where the drafter's argmax 3Di
  equals the ProstT5 greedy 3Di at that position.
- **mean k** = mean 3Di tokens generated per verifier (model) call = `mean_accepted + 1`.
  This is the theoretical speedup vs plain enc-dec (which does 1 token/call).

Scored **40/40** proteins, using the Colab-verified `refs_3di.json` (ProstT5 greedy
fold for every query, incl. `P0DTC9` which had no `query_3di.fasta` in the dataset).

## Headline

- Drafter is **strong**: ~0.90 accuracy prefix-blind, **0.93–0.94** with prefix context.
- **Prefix context `p` helps** (small but consistent): e.g. at K=15, mean_k 10.3 (static) → ~11.2 (p≥2).
- **More homologs is not better**: accuracy peaks around **N=16–32** and dips at N=64
  (the top-ranked homologs are closest to the query; distant ones dilute the consensus).
- **Best all-round config: N=32, prefix p=5.** mean_k ≈ 5.15 at K=5 (≈5× fewer model
  calls), rising to ~9 at K=11. At K=15 the small-N profile (N=8, p=4) edges ahead (11.26).

### Drafter accuracy — mode × #homologs (mean over 40 proteins)

| cfg    | N=4   | N=8   | N=16  | N=32  | N=64  |
|--------|-------|-------|-------|-------|-------|
| static | 0.910 | 0.913 | 0.910 | 0.902 | 0.898 |
| p=1    | 0.926 | 0.930 | 0.929 | 0.927 | 0.924 |
| p=2    | 0.929 | 0.934 | 0.934 | 0.934 | 0.930 |
| p=3    | 0.930 | 0.934 | 0.935 | 0.935 | 0.933 |
| p=4    | 0.931 | 0.934 | 0.935 | 0.936 | 0.933 |
| p=5    | 0.932 | 0.933 | 0.935 | 0.938 | 0.934 |

### mean k vs K (N=32, prefix p=5)

| K       | 1    | 3    | 5    | 8    | 11   | 15    |
|---------|------|------|------|------|------|-------|
| mean_k  | 1.93 | 3.61 | 5.15 | 7.20 | 9.02 | 11.19 |
| accept  | 0.93 | 0.88 | 0.85 | 0.80 | 0.76 | 0.71  |

(mean_k keeps rising with K; per-draft acceptance falls, as expected.)

## Files

- `drafter_accuracy.csv` — per (protein, N, mode, p): drafter accuracy.
- `mean_k_sweep.csv` — per (protein, N, mode, p, K): mean_accepted, mean_tokens_per_step
  (= mean k), acceptance_rate, n_steps.
- `summary_by_config.csv` — the above averaged across proteins.

## Reference verified (Colab T4, 2026-06-27)

`colab_verify_refs.py` regenerated ProstT5's greedy fold for every query and compared
to `query_3di.fasta`: **36/39 byte-for-byte identical; the other 3 differ by exactly
one 3Di token** (A0A6G0XC32 0.994, P0A9Q7 0.999, P62937 0.994 — the known fp16
tie-flip). So `query_3di` **is** ProstT5's greedy fold and **these numbers are
canonical**. No re-run needed.

The verify run also wrote `refs_3di.json` (uid -> greedy 3Di for all 40, incl. the
missing `P0DTC9`). To produce the complete 40-protein table with one self-consistent
reference:

    python3 folding_profile_sweep.py --data ../folding_MSA \
        --out folding_profile_sweep_results --refs refs_3di.json
