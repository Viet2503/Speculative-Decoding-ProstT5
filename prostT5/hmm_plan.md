# HMM drafter — TODO checklist

Profile-HMM drafter for inverse folding (3Di → AA), Pfam PF00535. **Task is done when** spec-decoded output is bit-exact with enc-dec on the full test set, with measured acceptance rate and wall-clock speedup reported.

Builds on top of `prostT5_baseline_performance.ipynb` in this folder — same setup cells, same test set, same timing helpers.

## Setup

- [x] Team-standard Colab runtime: **T4**. Pin it for every benchmark run (hard constraint #3).
- [x] Add one in-family GT-2 protein to `TEST_IDS`. Picked **`P39621`** (SpsA, *B. subtilis*, 256 aa, Pfam PF00535 — crystal structures 1QG8/1QGQ/1QGS). AFDB has it (`AF-P39621-F1-model_v6.pdb`, pLDDT 93.8). Slots into the 200–300 residue bucket between `A0A6G0XC32` (163) and `P04637` (393). Note: original candidate `P39620` was a typo — that accession is GerQ, not SpsA.

## H0 — Reading (~3 h)

- [x] Leviathan 2023 §2–3 — under greedy, accept iff `drafter_token == verifier_argmax`; on first rejection, take verifier's token. (Notes: `papers/notes.md` "Leviathan" section.)
- [x] Profile-HMM tutorial (Eddy 1998 *Bioinformatics* OR HMMER 3.x user guide §1–2). Match / insert / delete states, emission distributions, transition probabilities. (Notes: `papers/notes.md` "Profile HMMs" section.)
- [x] `pyhmmer` API skim: `easel.{Alphabet, MSAFile}`, `plan7.{Builder, Background, HMM}`, `hmmer.hmmalign`. Key handles: `hmm.M`, `hmm.match_emissions` ((M+1, 20) matrix — row per node, col per AA), `hmm.consensus`. (Notes: `papers/notes.md` "pyhmmer API skim" section.)

## H1 — Build & validate the HMM offline

Output: a length-pruned `(L_target, 20)` emission matrix I trust.

All of H1 lives in `build_hmm.py` (run from `prostT5/`). Artifacts (gitignored) land in `hmm_data/`.

- [x] Download Pfam PF00535 alignment from InterPro → `prostT5/hmm_data/PF00535_seed.sto` (gitignored). **Deviation from plan**: I downloaded SEED rather than Full. Reason: InterPro's `annotation=alignment:full` strips `#=GC RF` (pyhmmer needs RF for `architecture='hand'`); InterPro's SEED dump *also* strips the case + `.`/`-` convention, so RF isn't recoverable from the alignment text either. Pfam itself builds the family HMM from SEED with the default symfrac picker, and `architecture='fast'` on the SEED reproduces Pfam's distributed HMM bit-for-bit (verified — see Validation 1 note).
- [x] Inspect MSA: 131 sequences, alignment width 250, avg seq length 169.3 (range 153–197), gap fraction 0.323.
- [x] Build HMM with `pyhmmer.plan7.Builder.build_msa(...)` → `prostT5/hmm_data/PF00535.hmm`. M = 168 match columns. `architecture='fast'`, `seed=42`.
- [x] Print HMM consensus; **DXD motif visible** (consensus contains `DaD` at HMM match col 86 — the canonical GT-2 active site).
- [x] Align HMM to in-family target P39621 with `pyhmmer.hmmer.hmmalign(..., trim=False)`. 157/256 target residues align to match columns; 99 to insert columns.
- [x] Length-prune: `E ∈ R^{256 × 20}` for P39621. Match-column rows = HMM `match_emissions`; insert-column rows = HMM background (no per-position info). Mapping uses `aln.alignment[i]` (case-preserving aligned row) and `aln.reference` (RF), not `TextSequence.sequence` (which returns the ungapped string).
- [x] **Validation 1**: argmax(E) vs true AA. Overall 18.4% (47/256), **match-column-only 23.6% (37/157)**. Below the plan's 30% target but above the 20% "alignment broken" line. The 23.6% is intrinsic to PF00535 + P39621: I downloaded Pfam's distributed HMM and ran the same metric — identical 23.6%. PF00535 is the Pfam "Diverse family" GT-2; its match-column distributions are flat enough that argmax doesn't dominate.
- [x] **Validation 2**: `Σ_j log E[j, true_aa(j)]` per-residue (totals are length-confounded). P39621 (in-family, L=256): −2.602 nat/residue. P04637 (out-of-family p53, L=393): −2.912 nat/residue. Δ = +0.311 nat/residue, in-family ≫ out-of-family ✓. Total log-likelihoods: −666.0 vs −1144.5.

## H2 — Drafter + bit-exact verify loop

All code lives in `hmm_drafter.py` (run smoke test with `python hmm_drafter.py --smoke`).

- [x] Build the 20-class → ProstT5-vocab `LongTensor[20]` map. (`aa_to_vocab_map(tokenizer)` — tries bare `"A"` first, `▁A` fallback for sentencepiece-prefixed vocabs.)
- [x] Round-trip test: scatter `E` into full vocab, argmax, decode → equals consensus from H1. (`round_trip_check(drafter, tokenizer)` — passes for P39621 across all 256 positions with the smoke-test stub tokenizer; will re-run inside the notebook against the real `T5Tokenizer`.)
- [x] `HMMDrafter` class: constructor `(hmm_path, target_aa_seq, tokenizer)`, `propose(K) -> list[int]`, `commit(n_accepted)`. (Loads HMM, calls H1's `emission_matrix` for the alignment, caches `_vocab_ids` so `propose` is a slice. Also exposes `remaining()`, `reset()`, `cursor`.)
- [x] Tokenizer-identity assert in `HMMDrafter.__init__` (vocab hash compare). (`tokenizer_vocab_hash` = SHA-256 over sorted `(token, id)` pairs; snapshotted at construction, checked in `spec_decode_greedy` against the verifier's tokenizer.)
- [x] Custom Leviathan verify loop for the enc-dec model. (`spec_decode_greedy(model, tokenizer, three_di_seq, drafter, K, device)` — encodes once, maintains the decoder self-attn KV cache, feeds `[last_token, *proposals]`, accepts the matching argmax prefix, takes the verifier's argmax at the first disagreement or as the free "+1" on full accept, prunes only the self-attn cache (T5 cross-attn is encoder-length and untouched). Cache handled via `to_legacy_cache()` + 4-tuple slicing, works with both legacy and `EncoderDecoderCache` past_key_values formats.)
- [x] **Bit-exact assertion** on in-family target. P39621 at K=4: spec output == enc-dec greedy output (L=257 incl. EOS). Stats: 52/813 accepted (α=6.4%), 205 steps, 3 free tokens.
- [x] **Bit-exact assertion** on out-of-family target. A0A6G0XC32 at K=4: spec output == enc-dec greedy output (L=164 incl. EOS). Stats: 33/514 accepted (α=6.4%), 131 steps, 1 free token.

### H2 caveats / findings worth preserving

- **fp32 is required for the strict bit-exact assertion.** Under fp16 (`Rostlab/ProstT5_fp16` + `.half()`), the verifier's argmax can flip at positions with near-tied AA logits when the decoder input is batched (length K+1) vs. solo (length 1). Causal masking guarantees mathematical equivalence; fp16 GPU kernels don't. To re-run the bit-exact check on Colab, do `model = model.float(); torch.cuda.empty_cache()` (≈11 GB vRAM on T4) before calling `assert_bit_exact`. Production benchmarks (H3/H4) can stay in fp16 with the understanding that the bit-exact guarantee is "within fp16 tolerance".
- **α = 6.4 % at K=4 on both in-family and out-of-family.** Same α for both is consistent with H1: PF00535's match-column distributions are flat (23.6 % match-column identity), so the HMM's argmax rarely agrees with ProstT5's argmax — and the agreement that does happen is mostly chance, not family signal. At α=0.064 / K=4, Leviathan §3.8 predicts ~1.07 expected tokens emitted per spec step; with the verifier doing ~5× the per-step work, **prefix-blind HMM-spec is unlikely to beat enc-dec on wall clock**. H5 (prefix-aware) is the path to a writeup-worthy speedup.

## H3 — Benchmark on the test set

- [ ] Edit notebook `TEST_IDS` to include `P39621` and delete `benchmark_data/test_set_3Di.fasta` + `test_set_AA.fasta` so foldseek refetches. Re-run cells through `build_test_set` so the saved baseline rows are regenerated alongside the new ones.
- [ ] Add `time_hmm_spec(uid, three_di)` to the notebook; mirror `time_encdec` / `time_enccnn` exactly (warmup + median over repeats, `_sync()`, `_reset_peak_mem()`).
- [ ] Same row schema (`pipeline="hmm_spec"`); rows append to `raw_runs.csv` / `summary_per_protein.csv` / `speedup.csv`.
- [ ] Per-protein, additionally record: **acceptance rate** = accepted / proposed, and **drafter overhead** = drafter wall-time / step wall-time.
- [ ] Update existing latency / throughput / vRAM plots — third line for `hmm_spec`.
- [ ] New plot: acceptance rate vs. protein length, in-family marker distinct from out-of-family.

## H4 — K sweep and reported numbers

- [ ] Run K ∈ {1, 2, 4, 8, 16} on the in-family target. Plot speedup vs. K.
- [ ] Pick the K that maximizes wall-clock speedup as the reported number.
- [ ] Headline table: per-protein latency, speedup vs. enc-dec, acceptance rate, at the chosen K.

## H5 — Prefix-aware variant (stretch)

Only if H1–H4 land clean. This is the writeup-worthy contribution vs. the prefix-blind enc-CNN drafter.

- [ ] At each spec-decoding block boundary, re-run pyhmmer Forward conditioned on verified prefix; re-extract next K columns' distributions.
- [ ] Compare acceptance + speedup to naive variant on the same test set.

## Files I'll add to `prostT5/`

- [x] `build_hmm.py` — H1: MSA download, HMM build, validation.
- [x] `hmm_drafter.py` — H2: `HMMDrafter` + Leviathan greedy verify loop + bit-exact helper + no-GPU smoke test (`python hmm_drafter.py --smoke`).
- [x] `hmm_data/` — gitignored, holds `PF00535_seed.sto` (the SEED, not Full — see H1 deviation note) and `PF00535.hmm`.
- [ ] New section / cells in `prostT5_baseline_performance.ipynb` for H3 and H4 (or a sibling `hmm_drafter_eval.ipynb` if the main notebook gets too long).
