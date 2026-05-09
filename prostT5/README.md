# Baseline performance test — step-by-step

## Setup

1. **Pinned the env** to `transformers==4.46.3` + `protobuf>=3.20,<5` + `sentencepiece` (transformers 5.x has a regression that breaks `T5Tokenizer` loading from `spiece.model`, and ProstT5 ships only `spiece.model`).
2. **Loaded the slow `T5Tokenizer`** via `T5Tokenizer.from_pretrained('Rostlab/ProstT5_fp16', do_lower_case=False, legacy=True)` — the fast variant doesn't work because ProstT5's tokenizer is BPE, not Unigram.

## Models

1. **Loaded the enc–dec model** as `AutoModelForSeq2SeqLM.from_pretrained('Rostlab/ProstT5_fp16', torch_dtype=fp16)` and put it in `.eval().half()` on GPU.
2. **Reused the same encoder** for the enc–CNN path via `encoder = model.get_encoder()` — no second copy of ProstT5 in memory.
3. **Loaded the AA-CNN head** from `cnn_chkpnt_AA_CNN/model.pt` into a 4-tensor module that matches the checkpoint exactly: `Conv2d(1024→32, k=7) → ReLU → Dropout → Conv2d(32→20, k=7)` (234k params, 20 outputs = the 20 standard AAs in alphabetical order `ACDEFGHIKLMNPQRSTVWY`). Verified with `strict=True` load, no missing/unexpected keys.

## Test Sets

1. **Built the length-stratified test set** of 5 AlphaFoldDB proteins covering ~1 order of magnitude in length: **P01308 (110), A0A6G0XC32 (163), P04637 (393), P0DTC9 (419), P00533 (1 210)**.
2. **Resolved each PDB URL via the AFDB API** (`/api/prediction/{uid} → pdbUrl`) instead of hard-coding `model_v4.pdb` (AFDB has moved to `model_v6`; this also future-proofs against further bumps).
3. **Auto-installed Foldseek** for the runtime's OS (`foldseek-osx-universal` on Mac, `foldseek-linux-avx2` on Colab) into `prostT5/foldseek_bin/bin/`, made it executable.
4. **Extracted AA + 3Di per protein** by running `foldseek createdb` + `convert2fasta` on each downloaded PDB, then cached the result as `benchmark_data/test_set_AA.fasta` and `benchmark_data/test_set_3Di.fasta`.
5. **Defined the timing protocol** (identical for both pipelines):
  - **greedy, deterministic decoding** (`do_sample=False, num_beams=1`) for an apples-to-apples baseline,
    - **2 untimed warmup runs + 3 timed repeats** per protein, take the **median**,
    - `@torch.inference_mode()` + `model.eval()`, no autograd,
    - `torch.cuda.synchronize()` around every timed region,
    - `torch.cuda.reset_peak_memory_stats()` per protein → clean per-protein peak vRAM.

## Benchmarks

1. **Ran the enc–dec pipeline:** one `model.generate(...)` per protein, timing the entire autoregressive loop end-to-end.
2. **Ran the enc–CNN pipeline:** one encoder forward → trim `<fold2AA>` prefix and EOS from the hidden states (`h[:, 1:-1, :]`) → push through the AA-CNN → `argmax(-1)` → string of L AA letters.
3. **Recorded one row per `(protein, pipeline, repeat)`** with wall time, throughput (`generated_tokens / wall_s`), and peak vRAM.
4. **Aggregated and saved results** to `raw_runs.csv`, `summary_per_protein.csv`, and `speedup.csv`.
5. **Plotted latency / throughput / vRAM vs. protein length** on log scales, saved to `baseline_plots.png`.
6. **Read off the headline numbers:** enc–dec sits flat at ~21 tok/s (memory-bandwidth-bound autoregressive decoding), enc–CNN runs at 2 000–4 300 tok/s, raw wall-clock speedup is **98×–207×** across L = 110–1 210, peak vRAM differs by only ~0.1–0.4 GB (the gap is the enc–dec's KV cache).

---

## Future TODO — proper agreement / acceptance-rate (α) measurement

The pilot enc-dec ↔ enc-CNN per-residue identity number (~26%) we computed today is **not presentation-ready**. It needs to be redone in the drafter-integration phase (weeks 5–7) because:

1. It compares two argmax outputs under greedy; real α uses the verifier's full distribution via the speculative-sampling rule `min(1, p(y)/q(y))`, not just argmax identity.
2. The enc-CNN drafter is prefix-independent, so it doesn't reflect how a re-queriable drafter would behave inside an actual spec-decoding loop.
3. Sequence recovery against AFDB AA is also artificially low under greedy; ProstT5's published recovery uses sampling (`top_p=0.85, top_k=3, T=1.0, best-of-10`).
4. There's a 1-residue length mismatch on the longest protein (L=1 210 vs. 1 211 in the agreement table) — minor `<fold2AA>` / EOS slicing edge case to fix.

**What proper α-measurement looks like:** sweep enc-dec under greedy *and* sampling, compute α two ways (greedy-identity α, and Leviathan-rule sampling α), plug into Theorem 3.8 to predict wall-clock speedup at γ ∈ {3, 5, 8, 16}, then compare to **measured** speedup once the enc-CNN is wired into HuggingFace's `assistant_model` API.