# CLAUDE.md — `prostT5/`

This folder is the working directory for the **Profile-HMM drafter slice** of the team's Speculative-Decoding-ProstT5 project. Operational checklist lives in `hmm_plan.md`; this file states the framing and constraints that `hmm_plan.md` assumes.

## Project frame

TUM I12 (Chair of Bioinformatics) **PP1 SoSe2026**, 6-week course project. Goal: speed up ProstT5 enc-dec via Leviathan-style speculative decoding. The team splits the drafters (enc-CNN, Profile-HMM, 3Di-Flex profile); this slice owns the **Profile-HMM drafter, inverse-folding direction only (3Di → AA), Pfam PF00535** (Glycos_transf_2). The folding direction (AA → 3Di) is out of scope.

The chair's `project_description.pdf` is the spec. Re-read it before scope calls; do not expand scope.

## Hard constraints (from the assignment, not preference)

1. **Bit-exact equivalence** with greedy enc-dec output. Any divergence is a correctness bug — asserted in the H2 verify loop.
2. **Drafter and verifier must share tokenizer / vocab.** Mismatches break correctness silently — asserted on `HMMDrafter` init via a tokenizer hash.
3. **All benchmarks on identical hardware.** Pinned to Colab **T4** (16 GB, no bf16, no Ampere+-only kernels). Cross-drafter numbers only mean something on the fixed GPU.
4. **HMM must be length-pruned** to the target protein so columns align 1:1 with the template — otherwise acceptance rate collapses.

## Working style

- `hmm_plan.md` is the operational source of truth. Mark items as they ship; discovered work goes back into the plan, not into chat.
- Bit-exact is non-negotiable. If fp16 is required for benchmark speed but the bit-exact check needs fp32, run **both** and report each separately — don't pick one (see H2 caveat in `hmm_plan.md`).
- The team's `prostT5_baseline_performance.ipynb` is the shared notebook for evaluation. Extend it for H3/H4 cells; don't fork a sibling unless it gets too long.
