# HMM Summary 2

Source: `prostT5/hmm_summary-2.csv`

## Metric Meanings

- `bucket`: whether the protein belongs to the Pfam family used to build the HMM.
- `variant`: the drafter type.
  - `naive`: one-shot, prefix-blind HMM drafter.
  - `prefix_aware`: re-anchors to the verified prefix after each block.
- `k`: speculative block size, meaning how many draft tokens are proposed before verifier checking.
- `bit_exact`: fraction of proteins whose speculative decoding output exactly matches greedy enc-dec output.
- `accept_rate`: average fraction of proposed drafter tokens that the verifier accepts.
- `wall_s`: average end-to-end runtime in seconds per protein.
- `tokens_per_s`: average decoding throughput in tokens per second.
- `peak_vram_gb`: average peak GPU memory in GB.
- `reanchor_calls`: average number of HMM re-anchoring operations. This is `0` for the naive drafter.

## Average Comparison: In-Family vs Out-of-Family

### Naive Drafter

| K | In-family bit-exact | Out-of-family bit-exact | In-family accept | Out-of-family accept | In-family wall (s) | Out-of-family wall (s) | In-family tok/s | Out-of-family tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6667 | 1.0000 | 0.1714 | 0.1259 | 10.6551 | 10.4761 | 22.0494 | 21.9075 |
| 2 | 0.6667 | 1.0000 | 0.1034 | 0.0782 | 10.3487 | 10.2747 | 22.7040 | 22.9486 |
| 4 | 0.6667 | 1.0000 | 0.0593 | 0.0414 | 10.2606 | 10.4511 | 23.0286 | 21.9706 |
| 8 | 0.6667 | 1.0000 | 0.0308 | 0.0210 | 10.2832 | 10.2492 | 22.9169 | 22.8723 |
| 16 | 0.6667 | 1.0000 | 0.0161 | 0.0108 | 10.1125 | 10.1255 | 23.2702 | 23.4450 |

### Prefix-Aware Drafter

| K | In-family bit-exact | Out-of-family bit-exact | In-family accept | Out-of-family accept | In-family wall (s) | Out-of-family wall (s) | In-family tok/s | Out-of-family tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6667 | 1.0000 | 0.1621 | 0.1197 | 11.7595 | 11.2402 | 20.0292 | 20.0081 |
| 2 | 0.6667 | 1.0000 | 0.0985 | 0.0751 | 11.4422 | 10.9848 | 20.5321 | 21.4942 |
| 4 | 0.6667 | 1.0000 | 0.0568 | 0.0398 | 11.0912 | 10.6968 | 21.2000 | 21.3938 |
| 8 | 0.6667 | 1.0000 | 0.0295 | 0.0201 | 10.6336 | 10.9808 | 22.1732 | 21.0385 |
| 16 | 0.6667 | 1.0000 | 0.0154 | 0.0104 | 11.3242 | 10.4935 | 20.7535 | 22.2639 |

## Short Interpretation

- Out-of-family proteins have higher average `bit_exact` in this file: `1.0000` at every `K` for both drafter variants, versus `0.6667` for in-family.
- In-family proteins have consistently higher `accept_rate` than out-of-family for the same `variant` and `K`.
- The `naive` drafter is usually faster than `prefix_aware` in `wall_s` and usually slightly better in `tokens_per_s`.
- `prefix_aware` adds a substantial re-anchoring cost, so its latency is generally higher without a matching gain in exact-match rate in this file.