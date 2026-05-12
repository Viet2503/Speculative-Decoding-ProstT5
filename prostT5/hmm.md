# Profile-HMM Drafter

This repository slice implements a Profile-HMM drafter for speculative decoding with ProstT5. The scope is inverse folding only: given a 3Di sequence, generate amino acids (3Di -> AA). The HMM family currently targeted is Pfam `PF00535` (`Glycos_transf_2`), with `P39621` used as the in-family validation protein.

The high-level goal is to use a cheap Profile-HMM proposal model as the drafter and the full ProstT5 encoder-decoder as the verifier. Correctness is defined as bit-exact equality with greedy ProstT5 encoder-decoder generation.

## Repository Status

Implemented:

- `build_hmm.py`: downloads the Pfam SEED alignment, builds the PF00535 HMM, length-prunes emissions to target proteins, and runs offline validation.
- `hmm_drafter.py`: wraps the HMM emissions as a speculative drafter and implements a custom greedy Leviathan-style verify loop.
- `requirements-hmm.txt`: local dependencies for the HMM build and smoke test.
- `hmm_plan.md`: operational checklist and benchmark plan.

Not implemented yet:

- Notebook integration for full HMM timing on the shared benchmark set.
- K sweep over `K in {1, 2, 4, 8, 16}`.
- Final HMM latency / throughput / vRAM / acceptance-rate plots.
- Prefix-aware HMM variant.

## What The HMM Code Does

### H1: Build and validate a PF00535 Profile HMM

`build_hmm.py` performs the offline HMM construction work:

1. Downloads the Pfam `PF00535` SEED alignment from InterPro.
2. Reads the Stockholm MSA with `pyhmmer`.
3. Builds a Plan7 profile HMM with:
   - `pyhmmer.plan7.Builder`
   - `architecture="fast"`
   - `seed=42`
4. Writes the HMM to `hmm_data/PF00535.hmm`.
5. Fetches validation proteins from UniProt:
   - `P39621`: in-family GT-2 target, length 256
   - `P04637`: out-of-family p53 target, length 393
6. Aligns each target sequence to the HMM with `hmmalign`.
7. Converts the alignment into a length-pruned emission matrix `E` with shape:

```text
L_target x 20
```

Each row corresponds to one residue position in the target sequence. Columns follow the canonical amino-acid order:

```text
ACDEFGHIKLMNPQRSTVWY
```

For target residues aligned to HMM match columns, the row uses the HMM match-state emission distribution. For residues aligned to insert columns, the row uses the HMM background residue frequencies.

### H1 validation results recorded in the plan

The current validation numbers in `hmm_plan.md` are:

- PF00535 SEED MSA: 131 sequences, alignment width 250, average sequence length 169.3.
- Built HMM: `M = 168` match columns.
- The consensus contains the GT-2 `DXD` motif.
- For `P39621`, 157 of 256 residues align to match columns; 99 align to insert columns.
- Argmax identity vs true AA:
  - Overall: 18.4% (`47/256`)
  - Match-column only: 23.6% (`37/157`)
- True-AA log likelihood:
  - `P39621`: `-2.602` nat/residue
  - `P04637`: `-2.912` nat/residue
  - Delta: `+0.311` nat/residue in favor of the in-family protein

The key interpretation is that the HMM is aligned and family-aware, but its per-position argmax is weak. PF00535 is broad and its emission distributions are flat, so the HMM is not a strong direct AA predictor.

## HMM Drafter

`hmm_drafter.py` defines `HMMDrafter`.

At construction time, `HMMDrafter`:

1. Loads `hmm_data/PF00535.hmm`.
2. Aligns the target AA sequence to the HMM.
3. Builds the length-pruned emission matrix through `build_hmm.emission_matrix`.
4. Takes the argmax amino acid at each target position.
5. Maps those 20 amino-acid classes into ProstT5 tokenizer vocabulary IDs.
6. Caches the resulting vocab-ID sequence.

During decoding, the drafter exposes:

- `propose(K)`: returns the next `K` proposed token IDs.
- `commit(n_accepted)`: advances the drafter cursor.
- `remaining()`: reports how many target positions are left.
- `reset()`: returns the drafter to position zero.

The current drafter is prefix-blind. It proposes by absolute output position only and does not condition on the verified prefix. This is correct for speculative decoding because the verifier can reject bad proposals, but it limits the acceptance rate.

## Tokenizer Safety

The HMM drafter proposes ProstT5 vocab IDs, not raw amino-acid characters. To avoid silent correctness bugs, `hmm_drafter.py` hashes the tokenizer vocabulary:

```python
tokenizer_vocab_hash(tokenizer)
```

`HMMDrafter` snapshots the hash at construction time. `spec_decode_greedy` checks that the verifier tokenizer has the same hash before decoding. This enforces the project constraint that drafter and verifier share the same tokenizer / vocab mapping.

## Greedy Speculative Verify Loop

`spec_decode_greedy` implements a custom Leviathan-style greedy verification loop for the ProstT5 encoder-decoder model.

The loop:

1. Encodes the 3Di input once.
2. Maintains the decoder self-attention KV cache.
3. At each step, asks the HMM drafter for up to `K` proposed tokens.
4. Feeds `[last_token, *proposals]` through the verifier.
5. Accepts the longest proposal prefix whose tokens match the verifier argmax.
6. On first mismatch, emits the verifier argmax token.
7. On full acceptance, emits the verifier's extra "+1" token.
8. Prunes the self-attention KV cache to the verified output length.

The reference output is produced by `encdec_greedy_reference`, which mirrors the notebook's greedy encoder-decoder settings. `assert_bit_exact` compares speculative decoding against that reference and raises if any token differs.

## Current Bit-Exact Findings

The plan records successful bit-exact checks:

- `P39621`, `K=4`: spec output equals greedy encoder-decoder output.
  - Output length: 257 including EOS
  - Accepted: 52 / 813
  - Acceptance rate: 6.4%
  - Steps: 205
  - Free tokens: 3
- `A0A6G0XC32`, `K=4`: spec output equals greedy encoder-decoder output.
  - Output length: 164 including EOS
  - Accepted: 33 / 514
  - Acceptance rate: 6.4%
  - Steps: 131
  - Free tokens: 1

Important caveat: strict bit-exactness required fp32 in the recorded tests. In fp16, near-tied logits can flip between batched verifier calls and single-token greedy generation, even though the decoding logic is mathematically equivalent.

## Expected Performance

The measured acceptance rate so far is low: about 6.4% at `K=4`.

This is consistent with the weak HMM argmax identity observed in H1. Because the HMM proposals rarely match ProstT5's greedy argmax, the verifier rejects often. The plan notes that at `alpha = 0.064` and `K = 4`, the expected emitted tokens per speculative step are only about `1.07`. Since the verifier still does a larger batched step each iteration, this prefix-blind HMM drafter is unlikely to beat plain encoder-decoder generation on wall clock.

The likely path to a stronger result is H5: a prefix-aware HMM variant that recomputes or conditions proposal distributions on the verified prefix.

## How To Run

From `prostT5/`:

```bash
python3 -m venv .hmm_venv
source .hmm_venv/bin/activate
pip install -r requirements-hmm.txt
```

Build the HMM and validation artifacts:

```bash
python build_hmm.py
```

Run the no-GPU drafter smoke test:

```bash
python hmm_drafter.py --smoke
```

Generated artifacts are written to `hmm_data/`, which is gitignored and expected to be regenerated per machine.

## File Map

- `build_hmm.py`: HMM construction, emission extraction, validation.
- `hmm_drafter.py`: HMM drafter, tokenizer checks, speculative verify loop, smoke test.
- `hmm_plan.md`: checklist, recorded validation results, next benchmark tasks.
- `requirements-hmm.txt`: local HMM dependencies.
- `prostT5_baseline_performance.ipynb`: baseline benchmark notebook that still needs HMM benchmark cells.
- `README.md`: baseline ProstT5 encoder-decoder vs encoder-CNN benchmark notes.

