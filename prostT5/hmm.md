# Profile-HMM Drafter For ProstT5

This part of the project implements a profile-HMM drafter for ProstT5 in the
inverse-folding direction, `3Di -> AA`. The verifier is still the full
ProstT5 encoder-decoder. The drafter is a much cheaper family-aware model that
proposes amino-acid tokens in blocks before the verifier checks them.

The family we use is Pfam `PF00535` (`Glycos_transf_2`), which matches the
glycosyltransferase focus from the course description. The in-family target we
use for validation is `P39621` (SpsA from *B. subtilis*).

## Why a profile HMM helps here

Speculative decoding only helps if the drafter is both:

1. Cheaper than the verifier.
2. Correlated enough with the verifier that some proposals are accepted.

A profile HMM is useful because it gives us a family-level amino-acid prior at
each aligned position. Once we align the target protein to the HMM, we can
prune the family model down to the target length and obtain one amino-acid
distribution per residue position.

That gives us a drafter with three good properties:

1. It is cheap compared with running the full ProstT5 decoder.
2. It is biologically meaningful because it reflects conservation in the family
  MSA.
3. It can be made either prefix-blind or prefix-aware, which lets us compare a
  simple baseline against the more interesting adaptive variant.

## Overall workflow

The HMM path in this repository is split into two stages.

### 1. Build a family HMM and prune it to the target length

`build_hmm.py` does the offline biological preprocessing:

1. Download the Pfam `PF00535` SEED alignment from InterPro.
2. Build a Plan7 profile HMM with `pyhmmer.plan7.Builder`.
3. Align the target protein sequence to that HMM.
4. Read out a length-pruned emission matrix
  `E in R^{L_target x 20}`.

Each row of `E` corresponds to one target residue. The 20 columns follow the
canonical amino-acid order:

```text
ACDEFGHIKLMNPQRSTVWY
```

If the residue is aligned to an HMM match state, the row uses the match-state
emission distribution. If it lands in an insert state, we use the HMM
background distribution.

This gives us a family-aware, target-length-aligned prior over amino acids.

### 2. Use the pruned HMM as a speculative drafter

`hmm_drafter.py` converts those emissions into ProstT5 vocabulary proposals and
plugs them into a custom greedy Leviathan-style verification loop.

The key public surface is:

- `HMMDrafter`: naive prefix-blind drafter.
- `PrefixAwareHMMDrafter`: prefix-aware re-anchoring drafter.
- `spec_decode_greedy(...)`: speculative verification loop for ProstT5.
- `assert_bit_exact(...)`: checks speculative output against greedy enc-dec.

## The two drafter variants

### Naive variant: prefix-blind

The naive drafter is implemented by `HMMDrafter`.

It aligns the full target to the HMM once at construction time, computes the
length-pruned emission matrix once, maps its per-position amino-acid argmax to
ProstT5 vocabulary IDs, and then proposes by absolute output position.

That means:

- proposals are cheap,
- the target length stays fixed and aligned 1:1 with the template,
- but the drafter never reacts to what the verifier actually accepted.

So the naive variant is a good baseline, but it is stale after the first wrong
agreement between template and verified prefix.

### Prefix-aware variant: re-anchor on the verified prefix

The prefix-aware drafter is implemented by `PrefixAwareHMMDrafter`.

At each block boundary it:

1. Takes the amino-acid tokens that the verifier has already committed.
2. Replaces the corresponding prefix of the template sequence with those
  verified residues.
3. Re-aligns that hybrid sequence to the profile HMM.
4. Recomputes the remaining pruned emission rows for the suffix.

This is the HMM version of conditioning on the verified decoder state.

In short:

- naive drafter: proposals depend only on absolute position
- prefix-aware drafter: proposals depend on absolute position and the already
  verified prefix

The prefix-aware variant is more expensive because it realigns after each block,
but it is the more interesting experiment because it can adapt when the
verified prefix deviates from the one-shot template alignment.

## Why the prefix-aware version matters

The naive drafter mostly measures how much family conservation alone helps.
Its acceptance rate is limited by how often the static HMM proposal agrees with
ProstT5.

The prefix-aware variant adds the missing feedback loop:

1. Verified residues can change the effective alignment path.
2. A changed alignment path changes the next emissions.
3. Better next emissions can improve the next speculative block.

That makes the prefix-aware variant the more interesting answer to the project
prompt. It is closer to a true state-aware drafter rather than just a static
position-wise baseline.

## What the current code implements

### `build_hmm.py`

- Downloads the PF00535 SEED alignment.
- Builds the profile HMM.
- Aligns in-family and out-of-family targets.
- Produces the target-length-pruned emission matrix.
- Validates that the HMM is biologically sensible.

### `hmm_drafter.py`

- Builds the 20-class to ProstT5-vocab mapping.
- Implements the naive prefix-blind drafter.
- Implements the prefix-aware re-anchoring drafter.
- Implements the greedy speculative verification loop.
- Provides a smoke test that covers the drafter mechanics without needing a GPU.

### `prostT5_baseline_performance.ipynb`

The notebook is the evaluation surface. It is extended to:

1. Build or refresh the PF00535 HMM artifacts.
2. Assemble in-family and out-of-family evaluation proteins.
3. Run both HMM drafter variants.
4. Sweep over `K in {1, 2, 4, 8, 16}`.
5. Print only the important results:
  - bit-exact status
  - acceptance rate
  - latency
  - throughput
  - peak vRAM
6. Save HMM-specific CSV summaries for later analysis.

## Practical caveat

Strict bit-exact speculative verification is still most reliable in fp32.
Under fp16, near-tied logits can flip when the verifier processes a speculative
block instead of one token at a time. That caveat applies to both HMM
variants.

## Summary

What we are doing:

1. Build a glycosyltransferase profile HMM from a family MSA.
2. Prune it to the target protein's length so positions line up 1:1.
3. Use it as a cheap speculative drafter for ProstT5 inverse folding.
4. Compare a naive prefix-blind variant against a prefix-aware re-anchoring
  variant.

Why this helps:

1. The HMM is cheaper than the full verifier.
2. It injects a biologically meaningful family prior.
3. The naive variant gives a clear baseline.
4. The prefix-aware variant turns the HMM into a state-aware drafter, which is
  the more interesting candidate for improved acceptance and speedup.


