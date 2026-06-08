# ProstT5 Speculative Decoding — Notebook Overview

## High-Level Summary

This project benchmarks **speculative decoding** for [ProstT5](https://github.com/mheinzinger/ProstT5), a protein language model that translates between amino-acid sequences (AA) and 3Di structure tokens in both directions.

The standard decoder is autoregressive and slow — it generates one token at a time, taking O(L) sequential steps for a protein of length L. The key idea is to use a tiny **CNN head** as a fast *drafter*: it predicts all L tokens in a single parallel forward pass from the encoder hidden states. In speculative decoding, the CNN proposes a block of K tokens and the full T5 decoder verifies them in one step. If the drafter is accurate enough (high acceptance rate α), the speedup over plain autoregressive decoding can be substantial.

Two prediction directions are studied:

| Direction | Input | Output | Pipelines |
|-----------|-------|--------|-----------|
| **Folding** | AA sequence | 3Di structure tokens | `enc_dec_folding`, `enc_cnn_folding` |
| **Inverse folding** | 3Di structure tokens | AA sequence | `enc_dec`, `enc_cnn` |

All experiments run in **Google Colab** with a GPU, reading inputs and writing results to `MyDrive/models/` on Google Drive. The notebooks are designed to be run in order: dataset → baseline → speculative decoding.

---

## Notebooks

### 1. `100_protein_dataset.ipynb` — Dataset Builder

**Run this first, once.**

Downloads 100 protein structures from [AlphaFoldDB](https://alphafold.ebi.ac.uk/), extracts their 3Di structure tokens using [Foldseek](https://github.com/steineggerlab/foldseek), and writes two FASTA files to Google Drive.

**What it does:**
- Downloads PDB structures for a curated list of 100 proteins (stratified by length, taxonomy, and fold class), with 10 backup proteins in case any download fails
- Installs Foldseek from source (auto-detects Linux/macOS binary)
- Runs Foldseek's `easy-search` to extract per-residue 3Di tokens for each structure
- Writes `test_set_AA.fasta` (amino acid sequences) and `test_set_3Di.fasta` (3Di token sequences) to `MyDrive/models/`

**Outputs on Drive:**
```
MyDrive/models/
  test_set_AA.fasta
  test_set_3Di.fasta
```

---

### 2. `prostT5_baseline_performance.ipynb` — Baseline Benchmark

**Run after the dataset notebook.**

Times the two non-speculative pipelines (`enc_dec` and `enc_cnn`) on all 100 proteins in both directions and produces plots and sequence recovery statistics.

**Key design choices:**
- The **encoder runs once** per protein per direction; its hidden states are cached and reused for both the decoder timing pass and the CNN timing pass — no redundant encoder computation
- Each measurement is repeated **5 times**; the median is reported and the standard deviation is stored for error bars
- Per-protein checkpointing to Drive so a disconnected Colab session can resume

**What it benchmarks:**

| Pipeline | What runs | Latency dominated by |
|----------|-----------|----------------------|
| `enc_dec` | Encoder + autoregressive decoder | Decoder (O(L) sequential steps) |
| `enc_cnn` | Encoder + CNN head (1 forward pass) | Encoder |
| `enc_dec_folding` | Same as above in folding direction | Decoder |
| `enc_cnn_folding` | Same as above in folding direction | Encoder |

**Plots produced (all saved to Drive):**

- `baseline_plots.png` — all 4 pipelines combined: latency, throughput, CNN speedup for both directions
- `inv_folding_plots.png` — inverse folding only: latency, throughput, CNN speedup (error bar suppressed for shortest protein), sequence recovery
- `folding_plots.png` — folding only: same layout
- `combined_speedup.png` — both CNN speedup curves side-by-side with aligned y-axes

All plots use connected lines with shaded ±std bands. Axes show plain numbers (not scientific notation).

**Outputs on Drive:**
```
MyDrive/models/
  summary_per_protein.csv          (root copy — read by spec-dec notebooks)
  benchmark_results/
    summary_per_protein.csv
    raw_runs.csv
    sequence_recovery.csv
    sequence_recovery_summary.txt
    baseline_plots.png
    inv_folding_plots.png
    folding_plots.png
    combined_speedup.png
    baseline_checkpoint.pkl
```

---

### 3. `prostT5_spec_dec_folding_CNN.ipynb` — Speculative Decoding: Folding (AA → 3Di)

**Requires:** baseline notebook completed first (reads `summary_per_protein.csv` for enc-dec reference timings).

Benchmarks HuggingFace-native speculative decoding for the **folding direction** (AA → 3Di). The CNN head is wrapped as a `PreTrainedModel` (`CNNAssistantModel`) and passed directly to `model.generate(assistant_model=...)`, which implements the block-verify loop internally.

**Key concepts:**

- **Speculative decoding**: the CNN drafts K tokens in one parallel pass; the T5 decoder verifies the whole block in a single forward step. If all K tokens are accepted (match the greedy decoder), the next K tokens are drafted. Otherwise, generation falls back to the first rejected token. The output is **guaranteed identical** to plain greedy decoding (verified by sanity check on 5 proteins).
- **Acceptance rate α**: the per-position probability that the CNN drafter agrees with the T5 verifier. Higher α → more tokens accepted per step → higher speedup. Computed analytically and empirically.
- **K sweep**: benchmarked for K ∈ {1, 2, 4, 8} to find the optimal draft length.

**Sections:**
1. Configuration & model loading (ProstT5 + folding CNN checkpoint `CNN_fromAA_to3Di.pt`)
2. Dataset loading + hyperparameters
3. `CNNAssistantModel` — HuggingFace-compatible wrapper around the CNN
4. Helper functions (`run_encdec`, `run_hf_assisted`) with per-run timing and std
5. Sanity check: HF-assisted output == plain enc-dec greedy output
6. Full K-sweep benchmark (resume-aware, checkpointed)
7. Acceptance rate α analysis
8. Load results for offline analysis (works without the model)
9. Tables: per-K summary, predicted vs measured tokens/step, sequence recovery
10. Plots: 6-panel main figure, α vs length, per-protein speedup scatter

**Outputs on Drive:**
```
MyDrive/models/folding_spec_dec_results/
  folding_spec_dec_results.csv
  folding_spec_dec_predictions.json
  folding_alpha_results.csv
  folding_spec_dec_checkpoint.pkl
  folding_spec_dec_plots.png
  folding_alpha_analysis.png
  folding_per_protein_speedup.png
```

---

### 4. `prostT5_spec_dec_invfoldingCNN.ipynb` — Speculative Decoding: Inverse Folding (3Di → AA)

**Requires:** baseline notebook completed first.

Same approach as the folding notebook but for the **inverse folding direction** (3Di → AA), using the `CNN_from3di_toAA.pt` checkpoint.

**Differences from the folding notebook:**
- Input tokens are 3Di (lowercase), output tokens are amino acids (uppercase)
- Uses `<fold2AA>` prefix instead of `<AA2fold>`
- Separate CNN checkpoint for 3Di → AA translation
- Sequence recovery metric is compared against ground-truth AA sequences

**Sections mirror the folding notebook** (configuration, dataset, `CNNAssistantModel`, helpers, sanity check, K-sweep benchmark, α analysis, load results, tables, plots).

**Outputs on Drive:**
```
MyDrive/models/invfolding_spec_dec_results/
  invfolding_spec_dec_results.csv
  invfolding_spec_dec_predictions.json
  invfolding_alpha_results.csv
  invfolding_spec_dec_checkpoint.pkl
  invfolding_spec_dec_plots.png
```

---

## Drive Folder Structure

```
MyDrive/models/
  test_set_AA.fasta                    ← from 100_protein_dataset.ipynb
  test_set_3Di.fasta                   ← from 100_protein_dataset.ipynb
  CNN_from3di_toAA.pt                  ← upload manually before running
  CNN_fromAA_to3Di.pt                  ← upload manually before running
  summary_per_protein.csv              ← root copy for spec-dec notebooks
  benchmark_results/                   ← from prostT5_baseline_performance.ipynb
  folding_spec_dec_results/            ← from prostT5_spec_dec_folding_CNN.ipynb
  invfolding_spec_dec_results/         ← from prostT5_spec_dec_invfoldingCNN.ipynb
```

## Recommended Run Order

```
1. 100_protein_dataset.ipynb
2. prostT5_baseline_performance.ipynb
3. prostT5_spec_dec_folding_CNN.ipynb      (can run in parallel with step 4)
4. prostT5_spec_dec_invfoldingCNN.ipynb
```
