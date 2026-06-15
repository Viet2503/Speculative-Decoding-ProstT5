# Benchmark Data

This note summarizes the paired AA and 3Di benchmark data used by the ProstT5 folding experiments.

## Location

- Dataset directory: `/Users/chencheng-lin/Desktop/Speculative-Decoding-ProstT5/prostT5/prostT5_benchmarks/benchmark_data`
- AA FASTA: `/Users/chencheng-lin/Desktop/Speculative-Decoding-ProstT5/prostT5/prostT5_benchmarks/benchmark_data/test_set_AA.fasta`
- 3Di FASTA: `/Users/chencheng-lin/Desktop/Speculative-Decoding-ProstT5/prostT5/prostT5_benchmarks/benchmark_data/test_set_3Di.fasta`

Each FASTA record is keyed by protein ID. The same protein ID should appear in both files.

## What The Files Contain

- `test_set_AA.fasta` contains amino-acid sequences.
- `test_set_3Di.fasta` contains Foldseek 3Di structural-alphabet sequences.
- For the folding task, ProstT5 is prompted with an AA sequence and generates a 3Di sequence:

```text
<AA2fold> A A ...
```

## Current Dataset Summary

Generated with:

```bash
python3 datasey_analysis.py --markdown benchmark_data.md
```

| Metric | Value |
| --- | ---: |
| AA proteins | 100 |
| 3Di proteins | 100 |
| Proteins in both files | 100 |
| Matched AA/3Di lengths | 100 |
| Length mismatches | 0 |
| Only in AA FASTA | 0 |
| Only in 3Di FASTA | 0 |

## Length Statistics

| Sequence type | n | min | median | mean | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| AA | 100 | 46 | 412.5 | 613.6 | 2554 |
| 3Di | 100 | 46 | 412.5 | 613.6 | 2554 |

## Why Length Matching Matters

The folding benchmark assumes a one-to-one residue alignment:

```text
AA position i -> 3Di token position i
```

Because 3Di is a per-residue structural alphabet, each valid benchmark pair should have exactly one 3Di token for each amino-acid residue. This is required for:

- building position-specific HMM drafter probabilities,
- setting the expected generation length,
- comparing assisted generation to the ProstT5 greedy reference,
- avoiding silent alignment errors.

## Per-Protein Lengths

Use the analysis script to print the full per-protein table:

```bash
python3 datasey_analysis.py
```

For a shorter preview:

```bash
python3 datasey_analysis.py --limit 20
```

To save the table:

```bash
python3 datasey_analysis.py --csv dataset_lengths.csv
```
