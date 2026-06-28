# Dynamic-K algorithm — proposal (Gabriel)

Task: *"Implement a dynamic K selection dependent on the prediction confidence …
use the same for all drafters, so we should decide on one."* (Jan) + *"only use the
`assistant_model` method from HF."*

## Recommendation

**Use HF's built-in `assistant_confidence_threshold` as the shared dynamic-K
algorithm.** Each step, the drafter keeps proposing tokens while its top-1
probability ≥ τ, and stops at the first token below τ (capped at
`num_assistant_tokens`). This is *exactly* "dynamic K dependent on prediction
confidence", it's already in the `assistant_model` path (nothing custom to write),
and it's drafter-agnostic — every drafter exposes a softmax, so the same knob works
for CNN, FlexProfile and HMM.

```python
gen_cfg.num_assistant_tokens           = 16        # K_max cap per step
gen_cfg.num_assistant_tokens_schedule  = "constant"
gen_cfg.assistant_confidence_threshold = 0.5       # τ — the dynamic-K knob (tune per drafter)
out = model.generate(**inputs, assistant_model=assistant, generation_config=gen_cfg)
```

For this to behave, each drafter's assistant wrapper must emit **calibrated logits**
(= log-probabilities), so HF's confidence = the drafter's real top-1 probability.

## Evidence (folding profile drafter, 40 proteins, offline)

`folding_dynamic_k.py` simulates three draft-length policies (greedy spec-decode is
bit-identical to enc-dec, so this is exact, no GPU). Axes: mean k (tokens per model
call, ↑ = fewer verifier passes) vs drafter tokens proposed/step (∝ drafter
overhead). See `dynamic_k_comparison.png`.

| policy (static drafter) | mean k | proposed/step | draft acceptance |
|-------------------------|:------:|:-------------:|:----------------:|
| fixed K=5               | 4.74   | 4.9           | 0.76             |
| **confidence τ=0.7**    | **4.76** | **3.9**     | **0.96**         |
| fixed K=8               | 6.39   | 7.8           | 0.70             |
| **confidence τ=0.6**    | **6.25** | **5.7**     | **0.89**         |
| acceptance heuristic    | 8.28   | 9.9           | 0.68             |

- **Confidence-threshold Pareto-dominates fixed K**: at the *same* mean k it proposes
  ~20–25% fewer drafts and accepts ~0.90+ vs ~0.70–0.76 (τ=0.7 matches fixed K=5's
  mean k at 3.9 vs 4.9 proposals/step; τ=0.6 matches fixed K=8 at 5.7 vs 7.8). It
  spends the draft budget only where the drafter is confident — which is where it gets
  accepted. (Prefix-aware p=5 shows the same ordering, smaller gap; e.g. τ=0.5 →
  mean k 5.85 at 5.0 proposals vs fixed K=5 → 5.15 at 4.9.)
- The **acceptance heuristic** (`num_assistant_tokens_schedule="heuristic"`, +2/−1)
  is confidence-blind — it lands on the fixed-K curve and tends to over-draft
  (proposes ~10/step). No efficiency gain.

## Why this is the right pick for *us* specifically

The efficiency axis (mean k per proposed token) only matters when the drafter has
**per-token overhead** — i.e. the prefix-aware HMM (re-evaluates pyhmmer per token)
and the autoregressive CNN. That's exactly where the team saw **prefix-aware speedup
< 1**: it over-drafts expensive tokens that get rejected. Confidence-thresholding
cuts those wasted proposals, so it's the policy that can pull prefix-aware back above
breakeven. For a near-free drafter (static profile) it costs nothing to just raise K,
but the *same* knob is safe there too — so one algorithm covers everyone.

## Design decision for the team

- **One algorithm, per-drafter τ.** The *algorithm* is shared; the *threshold value*
  is not portable, because confidence scale differs by drafter. The prefix-aware HMM
  has a **compressed confidence range** (sparse context counts + Dirichlet smoothing),
  so τ > 0.7 starves it (proposes 0 → collapses to plain enc-dec); its useful band is
  τ ≈ 0.4–0.6. The static profile tolerates higher τ. CNN/FlexProfile should each be
  swept once. Suggested starting points: HMM/profile **τ≈0.5**, CNN **τ≈0.4** (HF
  default), then a quick per-drafter sweep.
- Optionally combine with `num_assistant_tokens_schedule="heuristic"` for the K_max
  cap, but the confidence threshold is doing the real work.

## Reproduce

    python3 folding_dynamic_k.py --data ../folding_MSA \
        --refs folding_profile_sweep_results/refs_3di.json \
        --out folding_profile_sweep_results
