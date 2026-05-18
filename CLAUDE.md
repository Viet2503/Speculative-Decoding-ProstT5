# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository explores **speculative decoding** applied to **ProstT5**, a protein language model that translates 3Di structural alphabet sequences into amino acid (AA) sequences (inverse folding). The goal is to use a fast **enc-CNN drafter** to accelerate the slow autoregressive **enc-dec** pipeline.

Two pipelines are benchmarked:
- **enc-dec**: Full ProstT5 encoder-decoder (`Rostlab/ProstT5_fp16`), autoregressive greedy decoding (~21 tok/s)
- **enc-CNN**: ProstT5 encoder + a tiny CNN head (`cnn_chkpnt_AA_CNN/model.pt`, 234k params), single parallel pass (~2000-4300 tok/s, 98-207x speedup)

## Environment Setup

```bash
conda create -n prostt5 python=3.12
pip install "transformers==4.46.3" "protobuf>=3.20,<5" sentencepiece
pip install torch accelerate>=0.26.0
```

Key constraint: `transformers==4.46.3` is pinned because transformers 5.x breaks `T5Tokenizer` loading from ProstT5's `spiece.model`.

## Running the Benchmark

The primary code lives in `prostT5/prostT5_baseline_performance.ipynb`. Run it on a CUDA GPU (designed for Google Colab with T4/A100). The notebook:
1. Mounts Google Drive and caches HF models to avoid re-downloading on reconnect
2. Downloads AlphaFold structures for a 100-protein test set (defined in `prostT5/test_set_100.py`)
3. Installs Foldseek to extract 3Di sequences from PDB files
4. Benchmarks both pipelines (2 warmup + 3 timed repeats, median reported)
5. Computes sampling-based acceptance rate α across 4 temperature/sampling configs
6. Generates extended agreement metrics (confusion matrix, per-AA, positional analysis)
7. Checkpoints after every protein (resumes from where it left off on Colab disconnect)
8. Outputs CSVs + plots to `prostT5/benchmark_results/`

The CNN checkpoint (`cnn_chkpnt_AA_CNN/model.pt`, ~1 MB) must be uploaded to Colab manually or stored in Drive. All other models download automatically.

## Key Technical Details

- **Tokenizer**: Must use `T5Tokenizer` (slow), not fast tokenizer. Load with `do_lower_case=False, legacy=True`.
- **3Di prompt format**: `<fold2AA>` prefix + space-separated lowercase 3Di letters
- **Encoder reuse**: The enc-CNN path calls `model.get_encoder()` — no second copy of ProstT5 weights in memory.
- **CNN architecture**: `Conv2d(1024→32, k=7) → ReLU → Dropout → Conv2d(32→20, k=7)`. Input is encoder hidden states with prefix/EOS trimmed (`h[:, 1:-1, :]`). Output is 20-class logits (standard AAs in alphabetical order: `ACDEFGHIKLMNPQRSTVWY`).
- **AA vocabulary**: The CNN outputs 20 classes corresponding to `ACDEFGHIKLMNPQRSTVWY` (alphabetical).

## Test Set

The 100-protein test set (`prostT5/test_set_100.py`) is stratified across:
- Length (20 per bin: tiny/short/medium/long/very_long)
- Secondary structure, organism, MSA depth, function, disease association
- Designed for compatibility with future Profile HMM drafter (varying MSA depth)

## Project Roadmap

Current phase: extended benchmarking with α measurement. Next phases (weeks 5-7): wire the enc-CNN as a drafter into HuggingFace's `assistant_model` API for actual speculative decoding, compare predicted vs. measured speedups at draft lengths γ ∈ {3, 5, 8, 16}, and evaluate Profile HMM as alternative drafter.
