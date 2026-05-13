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

## How the HMM produces predictions

Before comparing the two drafters, it helps to be explicit about what the
profile HMM is providing.

A profile HMM is a family model built from a multiple sequence alignment. For
each aligned match position, it stores a probability distribution over the 20
amino acids. Informally, each column answers the question:

```text
"At this family position, how likely is A, C, D, ..., Y?"
```

After we align one target protein to the HMM, we prune that family model down
to the target length. The result is a matrix:

```text
E in R^{L_target x 20}
```

where each row corresponds to one target residue position and each column is an
amino-acid score or probability. The drafter then takes the argmax in each row
and maps that amino acid into the ProstT5 vocabulary.

So at a very high level, one HMM prediction step is:

```text
HMM row -> best amino acid -> ProstT5 token id -> proposed token
```

### Example: how the naive drafter computes a prediction

Suppose one pruned emission row looks like this:

```text
position 4 distribution:
A: 0.05  C: 0.02  D: 0.03  E: 0.04  F: 0.01
G: 0.06  H: 0.02  I: 0.07  K: 0.04  L: 0.41
M: 0.03  N: 0.02  P: 0.05  Q: 0.03  R: 0.03
S: 0.02  T: 0.02  V: 0.11  W: 0.01  Y: 0.01
```

The largest value is for `L`, so the naive drafter stores:

```text
position 4 -> L -> tokenizer id for L
```

If this was done for all positions and the one-shot argmax sequence became:

```text
A G T L V P N Y
```

then `propose(3)` starting from cursor position 4 would simply read out:

```text
L, V, P
```

No recomputation happens at proposal time. The naive drafter just slices the
precomputed argmax sequence.

### Example: how the prefix-aware drafter computes a prediction

The prefix-aware drafter starts from the same idea, but after each verified
block it rebuilds the effective HMM rows for the suffix.

Suppose the original one-shot template was:

```text
A G T L V P N Y
```

and the verifier has already confirmed:

```text
A G S
```

Then the prefix-aware drafter forms the hybrid sequence:

```text
A G S L V P N Y
```

and realigns that hybrid sequence to the profile HMM. After re-anchoring, the
distribution at position 4 may change. For example, it might become:

```text
position 4 distribution after re-anchoring:
A: 0.03  C: 0.01  D: 0.02  E: 0.03  F: 0.01
G: 0.04  H: 0.02  I: 0.34  K: 0.03  L: 0.18
M: 0.02  N: 0.03  P: 0.04  Q: 0.05  R: 0.03
S: 0.02  T: 0.02  V: 0.16  W: 0.01  Y: 0.01
```

Now the best residue is `I`, not `L`, so the updated suffix begins with:

```text
I, ...
```

That is the core difference:

- naive drafter: compute the HMM argmax sequence once, then keep slicing it
- prefix-aware drafter: recompute the suffix argmax sequence after each
  verified block

## Worked examples of the two drafters

The easiest way to see the difference is to imagine that the HMM has already
been pruned to a target of length 8 and we decode with `K = 3`.

### Example 1: both drafters at the first block

Suppose the one-shot HMM alignment gives these per-position argmax amino acids:

```text
position:   1 2 3 4 5 6 7 8
HMM argmax: A G T L V P N Y
```

At the start, neither drafter has any verified prefix yet, so both propose the
same first block:

```text
propose(3) -> A, G, T
```

If the verifier also accepts `A, G, T`, then both drafters advance their
cursor to position 4.

### Example 2: naive drafter after a disagreement

Now suppose the verifier only accepts the first two proposed residues and then
emits a different third residue:

```text
drafted:   A, G, T
verified:  A, G, S
```

The naive drafter does not care that the verified prefix ended in `S` instead
of `T`. It only advances by how many amino-acid tokens were committed and keeps
using the original one-shot argmax table.

So on the next block it still proposes from positions 4, 5, 6 of the original
alignment:

```text
next naive propose(3) -> L, V, P
```

This is why it is called prefix-blind: once initialized, its future proposals
depend only on absolute position, not on what the verifier actually accepted.

### Example 3: prefix-aware drafter after the same disagreement

Start from the same situation:

```text
original template: A G T L V P N Y
verified prefix:   A G S
```

The prefix-aware drafter replaces the template prefix with the verified one,
forming a hybrid sequence:

```text
hybrid sequence:   A G S L V P N Y
```

It then realigns that hybrid sequence to the HMM and recomputes the emission
matrix. After re-anchoring, the best suffix might change. For example, the new
per-position argmax could become:

```text
position:          1 2 3 4 5 6 7 8
re-anchored argmax A G S I V Q N Y
```

Now the next prefix-aware proposal is different:

```text
next prefix-aware propose(3) -> I, V, Q
```

That is the key distinction. The prefix-aware drafter conditions on the
accepted prefix by recomputing the alignment-dependent emissions before it
drafts the next block.

### Example 4: when both variants stay identical for several steps

If the verifier keeps accepting exactly what the HMM already prefers, then the
two drafters can behave identically for multiple blocks.

For example, if the verified sequence so far is exactly:

```text
A, G, T, L, V, P
```

then the hybrid sequence used by the prefix-aware drafter is still the same as
the original template prefix. In that case, re-anchoring may produce the same
suffix emissions, so both drafters may still propose:

```text
next propose(2) -> N, Y
```

So the prefix-aware drafter is not guaranteed to differ every time. It only
diverges from the naive one when the verified prefix changes the effective HMM
alignment or the resulting suffix emissions.

### Example 5: why the prefix-aware variant is more interesting

The naive drafter effectively says:

```text
"At residue position i, what does the one-shot HMM alignment like best?"
```

The prefix-aware drafter says:

```text
"Given what the verifier has already confirmed up to position i - 1,
what does the HMM like best now for the next block?"
```

That second question is closer to the project requirement of conditioning on
already-verified residues, which is why the prefix-aware drafter is the more
adaptive and more interesting variant.

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

## How bit-exact is computed

`bit_exact` is an end-to-end equality check against the normal greedy
encoder-decoder output. It is not the same metric as HMM accuracy, family
membership, or acceptance rate.

For each protein, drafter variant, and `K`, the benchmark first computes a
reference output with:

```text
encdec_greedy_reference(model, tokenizer, three_di_seq, device)
```

That function calls `model.generate(...)` with greedy settings:

```text
num_beams = 1
do_sample = False
max_length = L + 2
min_length = L + 1
```

It then strips the leading decoder-start token, so the reference list contains
only the generated amino-acid tokens plus the final EOS token.

The speculative path then runs:

```text
spec_decode_greedy(model, tokenizer, three_di_seq, drafter, K, device)
```

This function performs Leviathan-style greedy verification:

1. Ask the HMM drafter for up to `K` proposed token IDs.
2. Feed `[last_token, *proposals]` into the ProstT5 decoder.
3. Compare each proposal to the verifier argmax at the same position.
4. Accept the longest matching prefix.
5. Emit accepted proposals plus the verifier argmax at the first mismatch. If
  all proposals match, emit the verifier's free `+1` argmax token.
6. Prune the decoder self-attention cache to the verified output length.
7. Advance the drafter by the number of emitted amino-acid tokens.

The raw benchmark row stores:

```text
bit_exact = spec_ids == ref_ids
```

where both sides are Python lists of tokenizer IDs. This is a strict token-level
comparison, not a decoded-string comparison. One different token ID anywhere in
the sequence makes the whole row `False`.

The CSV summaries aggregate that boolean in two steps:

1. `hmm_summary.csv` groups repeated runs for one
  `(protein, bucket, variant, K)` and uses `min(bit_exact)`. This means a
  configuration is marked bit-exact only if every repeat was bit-exact.
2. `hmm_average_by_bucket.csv` computes `mean(bit_exact)` over proteins. This
  is the fraction of proteins in that bucket whose summarized bit-exact value
  is `True`.

This explains why the bucket number can look counter-intuitive. In-family
proteins can have higher acceptance rate while still having lower
`bit_exact_fraction`: acceptance counts local proposal agreements, while
bit-exact requires the entire generated token sequence to match the greedy
reference.

## Code inspection of bit-exact correctness

The computation is correct for the intended strict token-equality question.

The important parts line up:

- `spec_decode_greedy(...)` returns generated token IDs excluding the
  decoder-start token and including EOS.
- `encdec_greedy_reference(...)` also returns token IDs excluding the
  decoder-start token and including EOS.
- The notebook compares these two raw ID lists directly with
  `spec_ids == ref_ids`.
- The cache pruning logic keeps the decoder self-attention cache at the
  verified prefix length. After a partial accept, the next step has cache for
  all tokens before `last_token`, and `last_token` is fed as the next decoder
  input. After a full accept, the free `+1` token becomes `last_token`, while
  the cache contains the previous tokens through the accepted block.
- The drafter cursor advances by emitted amino-acid tokens, not only by accepted
  proposals. That is correct because even a verifier fallback token consumes an
  output position.

There are two caveats.

First, the benchmark is run in fp16. In exact arithmetic, the speculative block
verifier and greedy one-token-at-a-time reference should choose the same argmax
tokens. On GPU fp16 kernels, near-tied logits can flip when the decoder sees a
block `[last_token, *proposals]` instead of a single token. That makes strict
bit-exact a numerical stability check as much as a logic check. For a
non-negotiable correctness claim, rerun the bit-exact assertion in fp32.

Second, `encdec_greedy_reference(...)` uses `generate(..., min_length=L+1)`,
while `spec_decode_greedy(...)` mirrors this mainly by setting
`max_new_tokens=L+1`; it does not explicitly apply the same EOS-suppression
logits processor used by `generate`. In the saved benchmark, every decoded
prediction has `pred_len == length`, so there is no evidence of early EOS in
these runs. Still, if ProstT5 ever preferred EOS before the required length,
this would be a real mismatch source to fix by suppressing EOS in the custom
loop until the minimum length is reached.

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

