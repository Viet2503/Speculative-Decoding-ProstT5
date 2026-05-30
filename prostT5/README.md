# Baseline performance test — step-by-step

## Setup

1. **Pinned the env** to `transformers==4.46.3` + `protobuf>=3.20,<5` + `sentencepiece` (transformers 5.x has a regression that breaks `T5Tokenizer` loading from `spiece.model`, and ProstT5 ships only `spiece.model`).
2. **Loaded the slow `T5Tokenizer`** via `T5Tokenizer.from_pretrained('Rostlab/ProstT5_fp16', do_lower_case=False, legacy=True)` — the fast variant doesn't work because ProstT5's tokenizer is BPE, not Unigram.
3. **Google Drive caching**: HuggingFace model cache (`HF_HOME`) is redirected to Google Drive to avoid re-downloading ProstT5 (~5.64 GB) on Colab reconnects. Checkpoints also persist to Drive.

## Models

1. **Loaded the enc–dec model** as `AutoModelForSeq2SeqLM.from_pretrained('Rostlab/ProstT5_fp16', torch_dtype=fp16)` and put it in `.eval().half()` on GPU.
2. **Reused the same encoder** for the enc–CNN path via `encoder = model.get_encoder()` — no second copy of ProstT5 in memory.
3. **Loaded the AA-CNN head** from `cnn_chkpnt_AA_CNN/model.pt` into a 4-tensor module that matches the checkpoint exactly: `Conv2d(1024→32, k=7) → ReLU → Dropout → Conv2d(32→20, k=7)` (234k params, 20 outputs = the 20 standard AAs in alphabetical order `ACDEFGHIKLMNPQRSTVWY`). Verified with `strict=True` load, no missing/unexpected keys.

## Test Sets

1. **Built a 100-protein test set** (see `test_set_100.py`) stratified across:
   - **Length**: 20 proteins per bin (tiny 50-100, short 101-250, medium 251-500, long 501-1000, very long 1001-2500)
   - **Secondary structure**: all-alpha, all-beta, alpha/beta, coil-rich/disordered
   - **Organism**: human, model organisms, bacteria, archaea/other
   - **MSA depth**: deep, moderate, shallow (for future Profile HMM drafter compatibility)
   - **Other factors**: disease association, fold complexity, function, AA composition bias
2. **Resolved each PDB URL via the AFDB API** (`/api/prediction/{uid} → pdbUrl`) instead of hard-coding `model_v4.pdb` (AFDB has moved to `model_v6`; this also future-proofs against further bumps).
3. **Auto-installed Foldseek** for the runtime's OS (`foldseek-osx-universal` on Mac, `foldseek-linux-avx2` on Colab) into `prostT5/foldseek_bin/bin/`, made it executable.
4. **Extracted AA + 3Di per protein** by running `foldseek createdb` + `convert2fasta` on each downloaded PDB, then cached the result as `benchmark_data/test_set_AA.fasta` and `benchmark_data/test_set_3Di.fasta`.
5. **Defined the timing protocol** (identical for both pipelines):
  - **greedy, deterministic decoding** (`do_sample=False, num_beams=1`) for an apples-to-apples baseline,
    - **2 untimed warmup runs + 3 timed repeats** per protein, take the **median**,
    - `@torch.inference_mode()` + `model.eval()`, no autograd,
    - `torch.cuda.synchronize()` around every timed region,
    - `torch.cuda.reset_peak_memory_stats()` per protein → clean per-protein peak vRAM.
6. **Checkpointing**: state saved after every protein (survives Colab disconnects via Google Drive).

## Benchmarks

1. **Ran the enc–dec pipeline:** one `model.generate(...)` per protein, timing the entire autoregressive loop end-to-end.
2. **Ran the enc–CNN pipeline:** one encoder forward → trim `<fold2AA>` prefix and EOS from the hidden states (`h[:, 1:-1, :]`) → push through the AA-CNN → `argmax(-1)` → string of L AA letters.
3. **Recorded one row per `(protein, pipeline, repeat)`** with wall time, throughput (`generated_tokens / wall_s`), and peak vRAM.
4. **Aggregated and saved results** to `raw_runs.csv`, `summary_per_protein.csv`, and `speedup.csv`.
5. **Plotted latency / throughput / vRAM vs. protein length** on log scales, saved to `baseline_plots.png`.
6. **Read off the headline numbers:** enc–dec sits flat at ~21 tok/s (memory-bandwidth-bound autoregressive decoding), enc–CNN runs at 2 000–4 300 tok/s, raw wall-clock speedup is **98×–207×** across L = 110–1 210, peak vRAM differs by only ~0.1–0.4 GB (the gap is the enc–dec's KV cache).

---

## Agreement & Acceptance Rate (α) Measurement

### Greedy Agreement Metrics
- **Per-residue identity**: fraction of positions where enc-dec and enc-CNN argmax outputs match
- **Per-AA class metrics**: precision, recall, F1 for each of 20 amino acid types
- **Confusion matrix**: 20x20 (rows = enc-dec reference, cols = CNN prediction)
- **Positional analysis**: agreement binned by normalized position (N-term to C-term, 10 bins)

### Sampling-Based Acceptance Rate α
Proper α computed via Leviathan's speculative-sampling rule:

```
α_t = Σ_v min(p_t(v), q_t(v))
```

Where:
- `p_t` = enc-dec (verifier) distribution at position t, obtained via teacher-forced forward pass
- `q_t` = enc-CNN (drafter) distribution at position t (prefix-independent, single parallel pass)

Swept across 4 sampling configurations:
1. **Greedy** (T=1.0, no filtering) — baseline
2. **ProstT5 published** (T=1.0, top_k=3, top_p=0.85)
3. **Conservative** (T=0.5) — sharper distributions
4. **Exploratory** (T=1.5) — flatter distributions

Results are plugged into Theorem 3.8 to predict wall-clock speedup at draft lengths γ ∈ {3, 5, 8, 16}.

### Fixed Issues
- Length mismatch bug on long proteins (enc-dec output now truncated to expected length L)

---

## Speculative decoding notebooks

| Notebook | Direction | Drafter |
|----------|-----------|---------|
| `prostT5_baseline_performance.ipynb` | 3Di → AA | enc-CNN (one-shot baseline) |
| `prostT5_spec_dec_CNN.ipynb` | 3Di → AA | enc-CNN + spec-dec loop |
| `prostT5_speculative_decoding_viet.ipynb` | 3Di → AA | enc-CNN + KV-cache + HMM (optional) |
| `prostT5_spec_dec_FlexProfile.ipynb` | **AA → 3Di** | **ProtProfileMD** ([Lüth et al. 2026](https://doi.org/10.64898/2026.01.21.700698)) |

`flexprofile_drafter.py` loads `finnlueth/ProtProfileMD` (LoRA + profile head) and defines `FlexProfileAssistantModel` for HF `assistant_model`. The FlexProfile notebook runs one-shot PPM predictions, greedy spec-dec with KV-cache, K sweeps, and HF assisted generation; outputs go to `flexprofile_spec_decode_results/`.

## Future TODO — Integration Phase

**What remains for the drafter-integration phase (weeks 5–7):**
1. Wire the enc-CNN into HuggingFace's `assistant_model` API for actual speculative decoding
2. Compare **predicted** speedup (from α + Theorem 3.8) to **measured** speedup
3. Address that enc-CNN is prefix-independent — measure how this affects real acceptance in a spec-decoding loop
4. Evaluate Profile HMM drafter as an alternative (test set already designed for MSA depth diversity)
5. FlexProfile: optional HF `assistant_model` wrapper; compare profile KL / argmax to spec-dec acceptance