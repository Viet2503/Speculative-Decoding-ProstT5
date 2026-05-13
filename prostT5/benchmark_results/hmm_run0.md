# HMM Run 0

Source: `benchmark_results/hmm_raw_runs.csv` filtered to `repeat = 0`.

This file follows the same report shape as the HMM summary: metric definitions first, then bucket-level averages, overall averages, best operating points, and a short interpretation.

## Metric Meanings

- `bucket`: whether the protein belongs to the Pfam family used to build the HMM.
- `variant`: the drafter type.
- `naive`: one-shot, prefix-blind HMM drafter.
- `prefix_aware`: re-anchors to the verified prefix after each speculative block.
- `k`: speculative block size, meaning how many draft tokens are proposed before verifier checking.
- `bit_exact`: fraction of proteins whose speculative decoding output exactly matches greedy enc-dec output.
- `accept_rate`: average fraction of proposed drafter tokens that the verifier accepts.
- `wall_s`: average end-to-end runtime in seconds per protein.
- `tokens_per_s`: average decoding throughput in tokens per second.
- `peak_vram_gb`: average peak GPU memory in GB.
- `reanchor_calls`: average number of HMM re-anchoring operations. This is `0` for the naive drafter.

## Run Coverage

- Total raw rows for run 0: `200`
- Proteins: `20` (`10` in-family, `10` out-of-family)
- Variants: `naive`, `prefix_aware`
- K values: `1`, `2`, `4`, `8`, `16`

## Average Comparison: In-Family vs Out-of-Family

### Naive Drafter

| K | In-family bit-exact | Out-of-family bit-exact | In-family accept | Out-of-family accept | In-family wall (s) | Out-of-family wall (s) | In-family tok/s | Out-of-family tok/s | In-family reanchors | Out-of-family reanchors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4000 | 1.0000 | 0.2239 | 0.1413 | 14.5259 | 11.1349 | 23.1843 | 22.4737 | 0.0 | 0.0 |
| 2 | 0.5000 | 0.9000 | 0.1401 | 0.0872 | 13.7339 | 10.9744 | 24.3559 | 23.0765 | 0.0 | 0.0 |
| 4 | 0.6000 | 0.9000 | 0.0823 | 0.0461 | 13.3524 | 11.1304 | 25.0736 | 22.1780 | 0.0 | 0.0 |
| 8 | 0.6000 | 1.0000 | 0.0427 | 0.0235 | 13.2867 | 11.2972 | 25.3797 | 22.5333 | 0.0 | 0.0 |
| 16 | 0.6000 | 0.9000 | 0.0222 | 0.0121 | 13.2027 | 10.9793 | 25.5962 | 23.3034 | 0.0 | 0.0 |

### Prefix-Aware Drafter

| K | In-family bit-exact | Out-of-family bit-exact | In-family accept | Out-of-family accept | In-family wall (s) | Out-of-family wall (s) | In-family tok/s | Out-of-family tok/s | In-family reanchors | Out-of-family reanchors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4000 | 1.0000 | 0.2174 | 0.1444 | 16.5841 | 11.9896 | 20.3389 | 20.8328 | 272.8 | 215.9 |
| 2 | 0.5000 | 0.9000 | 0.1361 | 0.0891 | 16.3700 | 11.8191 | 20.6732 | 21.6400 | 261.3 | 211.9 |
| 4 | 0.7000 | 0.9000 | 0.0798 | 0.0470 | 15.5481 | 11.6421 | 21.7237 | 21.7697 | 253.1 | 210.9 |
| 8 | 0.6000 | 1.0000 | 0.0415 | 0.0240 | 15.5423 | 11.8333 | 21.8895 | 21.3847 | 251.3 | 210.8 |
| 16 | 0.6000 | 0.9000 | 0.0215 | 0.0123 | 15.5616 | 11.4820 | 21.8845 | 22.5401 | 250.0 | 210.9 |

## Overall Averages

| Variant | K | Proteins | Bit-exact | Accept rate | Wall (s) | Tok/s | Peak vRAM (GB) | Reanchors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive | 1 | 20 | 0.7000 | 0.1826 | 12.8304 | 22.8290 | 6.0137 | 0.0 |
| naive | 2 | 20 | 0.7000 | 0.1136 | 12.3541 | 23.7162 | 6.0139 | 0.0 |
| naive | 4 | 20 | 0.7500 | 0.0642 | 12.2414 | 23.6258 | 6.0142 | 0.0 |
| naive | 8 | 20 | 0.8000 | 0.0331 | 12.2919 | 23.9565 | 6.0148 | 0.0 |
| naive | 16 | 20 | 0.7500 | 0.0171 | 12.0910 | 24.4498 | 6.0160 | 0.0 |
| prefix_aware | 1 | 20 | 0.7000 | 0.1809 | 14.2868 | 20.5858 | 6.0137 | 244.3 |
| prefix_aware | 2 | 20 | 0.7000 | 0.1126 | 14.0945 | 21.1566 | 6.0139 | 236.6 |
| prefix_aware | 4 | 20 | 0.8000 | 0.0634 | 13.5951 | 21.7467 | 6.0142 | 232.0 |
| prefix_aware | 8 | 20 | 0.8000 | 0.0327 | 13.6878 | 21.6371 | 6.0148 | 231.1 |
| prefix_aware | 16 | 20 | 0.7500 | 0.0169 | 13.5218 | 22.2123 | 6.0160 | 230.4 |

## Best Operating Points

### Fastest Overall

- `naive`, `K=16`: `12.091 s`, bit-exact `0.7500`, throughput `24.45 tok/s`
- `naive`, `K=4`: `12.241 s`, bit-exact `0.7500`, throughput `23.63 tok/s`
- `naive`, `K=8`: `12.292 s`, bit-exact `0.8000`, throughput `23.96 tok/s`
- `naive`, `K=2`: `12.354 s`, bit-exact `0.7000`, throughput `23.72 tok/s`
- `naive`, `K=1`: `12.830 s`, bit-exact `0.7000`, throughput `22.83 tok/s`

### Highest Bit-Exact Fraction

- `naive`, `K=8`: bit-exact `0.8000`, wall `12.292 s`, accept `0.0331`
- `prefix_aware`, `K=4`: bit-exact `0.8000`, wall `13.595 s`, accept `0.0634`
- `prefix_aware`, `K=8`: bit-exact `0.8000`, wall `13.688 s`, accept `0.0327`
- `naive`, `K=16`: bit-exact `0.7500`, wall `12.091 s`, accept `0.0171`
- `naive`, `K=4`: bit-exact `0.7500`, wall `12.241 s`, accept `0.0642`

### Highest Acceptance Rate

- `naive`, `K=1`: accept `0.1826`, wall `12.830 s`, bit-exact `0.7000`
- `prefix_aware`, `K=1`: accept `0.1809`, wall `14.287 s`, bit-exact `0.7000`
- `naive`, `K=2`: accept `0.1136`, wall `12.354 s`, bit-exact `0.7000`
- `prefix_aware`, `K=2`: accept `0.1126`, wall `14.095 s`, bit-exact `0.7000`
- `naive`, `K=4`: accept `0.0642`, wall `12.241 s`, bit-exact `0.7500`

## Per-Protein Average By Variant

Each row averages the five K settings for one protein and one drafter variant.

| Protein | Bucket | Variant | Bit-exact | Accept rate | Wall (s) | Tok/s | Reanchors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| O45293 | in_family | naive | 0.8000 | 0.0701 | 18.5835 | 22.7278 | 0.0 |
| O45293 | in_family | prefix_aware | 0.8000 | 0.0672 | 22.1065 | 19.0944 | 353.0 |
| O60762 | in_family | naive | 0.8000 | 0.1193 | 10.0674 | 25.9953 | 0.0 |
| O60762 | in_family | prefix_aware | 0.8000 | 0.1193 | 11.7569 | 22.2631 | 190.6 |
| O70152 | in_family | naive | 0.0000 | 0.1241 | 9.9639 | 26.2881 | 0.0 |
| O70152 | in_family | prefix_aware | 0.0000 | 0.1241 | 11.6908 | 22.3507 | 188.8 |
| P39621 | in_family | naive | 0.2000 | 0.0852 | 10.9892 | 23.4223 | 0.0 |
| P39621 | in_family | prefix_aware | 0.2000 | 0.0763 | 12.5698 | 20.4519 | 209.2 |
| Q67FW5 | in_family | naive | 0.8000 | 0.0734 | 15.1440 | 22.9324 | 0.0 |
| Q67FW5 | in_family | prefix_aware | 0.8000 | 0.0734 | 17.7412 | 19.5775 | 285.2 |
| Q7Z4T8 | in_family | naive | 0.8000 | 0.0720 | 19.4590 | 22.8626 | 0.0 |
| Q7Z4T8 | in_family | prefix_aware | 0.8000 | 0.0727 | 23.5809 | 18.8462 | 365.0 |
| Q9D4M9 | in_family | naive | 0.6000 | 0.1068 | 17.4897 | 24.7667 | 0.0 |
| Q9D4M9 | in_family | prefix_aware | 0.6000 | 0.1068 | 21.4053 | 20.2045 | 325.0 |
| Q9LM93 | in_family | naive | 1.0000 | 0.1093 | 9.8289 | 25.1619 | 0.0 |
| Q9LM93 | in_family | prefix_aware | 1.0000 | 0.1093 | 11.1233 | 22.2678 | 186.0 |
| Q9VLQ1 | in_family | naive | 0.0000 | 0.1258 | 12.5233 | 26.1881 | 0.0 |
| Q9VLQ1 | in_family | prefix_aware | 0.0000 | 0.1222 | 13.5619 | 24.1508 | 237.6 |
| Q9Y673 | in_family | naive | 0.4000 | 0.1367 | 12.1543 | 26.8342 | 0.0 |
| Q9Y673 | in_family | prefix_aware | 0.6000 | 0.1213 | 13.6755 | 23.8124 | 236.6 |
| A0A6G0XC32 | out_of_family | naive | 1.0000 | 0.0952 | 6.8396 | 24.0401 | 0.0 |
| A0A6G0XC32 | out_of_family | prefix_aware | 1.0000 | 0.0879 | 7.2051 | 22.7781 | 131.4 |
| A0PK11 | out_of_family | naive | 1.0000 | 0.0815 | 9.7954 | 23.8295 | 0.0 |
| A0PK11 | out_of_family | prefix_aware | 1.0000 | 0.0791 | 10.2201 | 22.8221 | 188.8 |
| A1A519 | out_of_family | naive | 1.0000 | 0.0075 | 16.9724 | 19.5322 | 0.0 |
| A1A519 | out_of_family | prefix_aware | 1.0000 | 0.0075 | 17.7804 | 18.6217 | 322.4 |
| A1L190 | out_of_family | naive | 1.0000 | 0.1305 | 3.3730 | 26.5062 | 0.0 |
| A1L190 | out_of_family | prefix_aware | 1.0000 | 0.1554 | 3.3407 | 26.7425 | 60.8 |
| A1L3X0 | out_of_family | naive | 1.0000 | 0.0439 | 13.0589 | 21.5948 | 0.0 |
| A1L3X0 | out_of_family | prefix_aware | 1.0000 | 0.0421 | 13.9454 | 20.2218 | 252.2 |
| A2RU14 | out_of_family | naive | 1.0000 | 0.0964 | 4.6309 | 25.1925 | 0.0 |
| A2RU14 | out_of_family | prefix_aware | 1.0000 | 0.0964 | 4.9543 | 23.6830 | 89.0 |
| A2RUC4 | out_of_family | naive | 0.4000 | 0.0493 | 14.5879 | 21.6778 | 0.0 |
| A2RUC4 | out_of_family | prefix_aware | 0.4000 | 0.0493 | 15.4977 | 20.3907 | 278.4 |
| A4GXA9 | out_of_family | naive | 1.0000 | 0.0447 | 17.4390 | 21.7956 | 0.0 |
| A4GXA9 | out_of_family | prefix_aware | 1.0000 | 0.0447 | 18.7387 | 20.2888 | 335.0 |
| P01308 | out_of_family | naive | 1.0000 | 0.0476 | 4.9373 | 22.6436 | 0.0 |
| P01308 | out_of_family | prefix_aware | 1.0000 | 0.0476 | 5.1497 | 21.7395 | 94.8 |
| P04637 | out_of_family | naive | 1.0000 | 0.0235 | 19.3982 | 20.3174 | 0.0 |
| P04637 | out_of_family | prefix_aware | 1.0000 | 0.0235 | 20.7000 | 19.0465 | 368.0 |

## Short Interpretation

- The fastest overall setting is `naive`, `K=16` with mean wall time `12.091 s` and bit-exact fraction `0.7500`.
- The fastest prefix-aware setting is `K=16` with mean wall time `13.522 s`; it remains slower than the fastest naive setting because of re-anchoring overhead.
- Acceptance rate is highest at `K=1` for both variants: naive `0.1826`, prefix-aware `0.1809`. Acceptance falls as K grows because more draft tokens are proposed per verification step.
- Out-of-family proteins are faster and more often bit-exact than in-family proteins in this run, even though in-family proteins usually have higher acceptance rates.
- Prefix-aware re-anchoring slightly improves bit-exact fraction at `K=4` overall, but it does not improve runtime in run 0.

### Best In-Family Averages By Wall Time

- `naive`, `K=16`: `13.203 s`, bit-exact `0.6000`, accept `0.0222`
- `naive`, `K=8`: `13.287 s`, bit-exact `0.6000`, accept `0.0427`
- `naive`, `K=4`: `13.352 s`, bit-exact `0.6000`, accept `0.0823`

### Best Out-of-Family Averages By Wall Time

- `naive`, `K=2`: `10.974 s`, bit-exact `0.9000`, accept `0.0872`
- `naive`, `K=16`: `10.979 s`, bit-exact `0.9000`, accept `0.0121`
- `naive`, `K=4`: `11.130 s`, bit-exact `0.9000`, accept `0.0461`

