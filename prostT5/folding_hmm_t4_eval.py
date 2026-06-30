"""CUDA/T4 folding-HMM benchmark over prostT5/prostT5_benchmarks/folding_MSA.

This is the real runtime pipeline. It refuses to run without CUDA, uses only the
local prostT5_benchmarks/folding_MSA directory, and writes resumable per-protein rows under
results/folding_hmm_results/eval_result.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, GenerationConfig, PreTrainedModel, T5Config, T5Tokenizer
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from folding_profile_sweep import (
    FoldingProfileDrafter,
    GAP_CHARS,
    K_VALUES,
    N_HOMOLOG_VALUES,
    P_VALUES,
    load_protein,
    read_fasta,
    simulate_fixed_k,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "prostT5_benchmarks" / "folding_MSA"
DEFAULT_OUT = SCRIPT_DIR / "results" / "folding_hmm_results" / "eval_result"
EXECUTION_MODE = "t4_cuda_prostt5"
MODEL_NAME = "Rostlab/ProstT5_fp16"
COMPLETION_FIELDS = ("protein_id", "mode", "p", "K", "n_homologs_requested", "execution_mode")
CONFIGS_PER_N = len(K_VALUES) * (1 + len(P_VALUES))
print = functools.partial(print, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--filename", default="folding_hmm_eval_result.csv")
    parser.add_argument("--benchmark-fasta", default=None,
                        help="optional FASTA whose record IDs define the proteins to evaluate, e.g. test_set_AA_2.fasta")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--protein-limit", type=int, default=None)
    parser.add_argument("--max-configs", type=int, default=None,
                        help="optional smoke-test cap on new non-benchmark configs")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="print a text progress line after this many new configs")
    parser.add_argument("--k-values", default=None,
                        help="comma-separated K values; default uses the full grid")
    parser.add_argument("--p-values", default=None,
                        help="comma-separated prefix p values; default uses the full grid; use 'none' for static-only")
    parser.add_argument("--n-values", default=None,
                        help="comma-separated homolog-count values; default uses the full grid")
    parser.add_argument("--min-homologs", type=int, default=0,
                        help="exclude proteins with fewer projected homolog rows before running configs")
    parser.add_argument("--include-static", action="store_true",
                        help="also run static HMM rows for the selected K/N values")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="do not run pure ProstT5 baseline rows; speedup_runtime will be blank for new assisted rows")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark must run on a CUDA GPU, e.g. Colab T4. Current torch.cuda.is_available() is False.")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True

    data = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename
    completed_path = out_dir / "folding_hmm_eval_completed_keys.csv"
    summary_path = out_dir / "folding_hmm_eval_run_summary.csv"

    existing_rows = [] if args.force else [
        _normalize_row(r) for r in _read_rows(out_path)
        if r.get("execution_mode") == EXECUTION_MODE
    ]
    completed = {_completion_key(r) for r in existing_rows if _has_key(r)}
    print(f"Resume: {len(existing_rows)} existing T4 rows, {len(completed)} completed keys")

    valid_dirs = sorted(
        p for p in data.iterdir()
        if p.is_dir() and (p / "homologs_projected_to_query_3di.fasta").exists()
    )
    requested_ids = _read_id_filter(args.benchmark_fasta)
    if requested_ids is not None:
        before = len(valid_dirs)
        valid_dirs = [p for p in valid_dirs if p.name in requested_ids]
        missing_ids = sorted(requested_ids - {p.name for p in valid_dirs})
        print(
            f"Benchmark FASTA filter: {len(valid_dirs)}/{before} MSA folders match "
            f"{args.benchmark_fasta}"
        )
        if missing_ids:
            print("Missing MSA folders for benchmark IDs: " + ", ".join(missing_ids[:30]))
            if len(missing_ids) > 30:
                print(f"... plus {len(missing_ids) - 30} more missing IDs")
    if args.min_homologs:
        before = len(valid_dirs)
        valid_dirs = [
            p for p in valid_dirs
            if _count_fasta_records(p / "homologs_projected_to_query_3di.fasta") >= args.min_homologs
        ]
        print(
            f"Min-homolog filter: {len(valid_dirs)}/{before} MSA folders have "
            f">= {args.min_homologs} projected homologs"
        )
    loaded = [load_protein(d) for d in valid_dirs]
    if args.protein_limit is not None:
        loaded = loaded[:args.protein_limit]
    k_values = _parse_int_list(args.k_values, K_VALUES)
    p_values = _parse_int_list(args.p_values, P_VALUES)
    n_values = _parse_int_list(args.n_values, N_HOMOLOG_VALUES)

    sym_to_idx = _build_alphabet(loaded)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, do_lower_case=False, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).to(device).eval()
    model = model.half()

    three_di_alphabet = "".join(sym_to_idx.keys())
    three_di_token_ids = [tokenizer.encode(f" {tok}", add_special_tokens=False)[0] for tok in three_di_alphabet]
    token_id_to_3di = {tid: tok for tok, tid in zip(three_di_alphabet, three_di_token_ids)}
    assistant = FoldingProfileAssistant(
        model.config,
        model.get_encoder(),
        three_di_token_ids,
        token_id_to_3di,
        device,
    ).to(device).eval()

    new_rows: list[dict] = []
    skipped_no_ref: list[str] = []
    short_homologs: list[str] = []
    new_configs = 0
    total_remaining = _count_remaining_configs(
        loaded, data, completed, k_values, p_values, n_values, args.include_static
    )
    if args.max_configs is not None:
        total_remaining = min(total_remaining, args.max_configs)
    selected_configs_per_n = len(k_values) * (len(p_values) + (1 if args.include_static else 0))
    print(
        f"Benchmark plan: {len(loaded)} proteins, {selected_configs_per_n} selected configs per N, "
        f"selected K={k_values}, p={p_values}, N={n_values}, include_static={args.include_static}, "
        f"remaining assisted configs={total_remaining}"
    )

    progress = _make_progress(total_remaining, desc="T4 folding HMM configs")
    started_at = time.perf_counter()

    for protein_i, (uid, q3, projected_rows) in enumerate(loaded, start=1):
        aa_path = data / uid / "query_aa.fasta"
        if q3 is None or not aa_path.exists():
            skipped_no_ref.append(uid)
            continue
        aa = read_fasta(aa_path)[0][1].upper()
        length = len(aa)
        ref = list(q3)
        ref_idx = np.array([sym_to_idx.get(c, -1) for c in ref])
        homologs_avail = len(projected_rows)
        protein_new_start = new_configs
        print(f"\n[{protein_i}/{len(loaded)}] {uid} L={length} homologs={homologs_avail}")

        bench_key = _make_key(uid, "benchmark", "", "", "")
        baseline = None
        if not args.skip_baseline and bench_key not in completed:
            print(f"  {uid}: running pure ProstT5 baseline")
            baseline = run_generation(model, tokenizer, aa, device, assistant=None)
            new_rows.append(_row(
                uid=uid, protein_i=protein_i, length=length,
                pipeline="prostT5_benchmark", mode="benchmark", p="", k="",
                n_requested="", n_used="", n_avail=homologs_avail,
                timings=baseline, seq=baseline["seq"], ref=q3,
                mean_k=1.0, acceptance_rate=1.0, mean_accepted=0.0,
                n_steps=length, drafter_accuracy="", speedup_runtime=1.0,
            ))
            _flush_checkpoints(out_path, completed_path, existing_rows, new_rows)
        else:
            baseline = None

        baseline_wall = _existing_baseline_wall(existing_rows, uid)
        if baseline_wall is None and baseline is not None:
            baseline_wall = baseline["wall_s"]
        if baseline_wall is None and not args.skip_baseline:
            print(f"  {uid}: baseline already completed but wall_s was not found; runtime speedup will be blank for new rows")

        for n_requested in n_values:
            rows_n = projected_rows[:n_requested]
            n_used = len(rows_n)
            if homologs_avail < n_requested:
                msg = f'protein "{uid}" has only {homologs_avail} homologs, requires {n_requested}; using {n_used}.'
                print(msg)
                short_homologs.append(msg)

            build_t0 = time.perf_counter()
            drafter = FoldingProfileDrafter(rows_n, len(ref), sym_to_idx)
            drafter_build_s = time.perf_counter() - build_t0

            configs = []
            if args.include_static:
                configs.extend(("static", "", k) for k in k_values)
            configs.extend(("prefix", p, k) for p in p_values for k in k_values)
            for mode, p_val, k in configs:
                key = _make_key(uid, mode, p_val, k, n_requested)
                if key in completed:
                    continue
                if args.max_configs is not None and new_configs >= args.max_configs:
                    break

                config_t0 = time.perf_counter()
                match, drafter_accuracy = _match_mask(drafter, ref, ref_idx, mode, p_val)
                steps, mean_accepted, mean_k, acceptance_rate = simulate_fixed_k(match, int(k))

                assistant.set_active(drafter, mode=mode, p=int(p_val) if p_val != "" else 0)
                assistant.generation_config.num_assistant_tokens = float(k)
                try:
                    timings = run_generation(model, tokenizer, aa, device, assistant=assistant)
                    speedup_runtime = baseline_wall / timings["wall_s"] if baseline_wall and timings["wall_s"] > 0 else ""
                    seq = timings["seq"]
                    eval_status = "ok"
                    error_message = ""
                except torch.OutOfMemoryError as exc:
                    print(f"  {uid}: CUDA OOM for mode={mode} p={p_val} K={k} N={n_requested}; recording oom row and continuing")
                    _clear_cuda()
                    timings = _empty_timings()
                    speedup_runtime = ""
                    seq = ""
                    eval_status = "oom"
                    error_message = _short_error(exc)
                new_rows.append(_row(
                    uid=uid, protein_i=protein_i, length=length,
                    pipeline="folding_hmm_assisted", mode=mode, p=p_val, k=k,
                    n_requested=n_requested, n_used=n_used, n_avail=homologs_avail,
                    timings=timings, seq=seq, ref=q3,
                    mean_k=mean_k, acceptance_rate=acceptance_rate,
                    mean_accepted=mean_accepted, n_steps=steps,
                    drafter_accuracy=drafter_accuracy,
                    speedup_runtime=speedup_runtime,
                    drafter_build_s=drafter_build_s,
                    eval_status=eval_status,
                    error_message=error_message,
                ))
                _flush_checkpoints(out_path, completed_path, existing_rows, new_rows)
                new_configs += 1
                elapsed_config = time.perf_counter() - config_t0
                progress.update(1)
                if args.progress_every and (new_configs % args.progress_every == 0):
                    elapsed_total = time.perf_counter() - started_at
                    wall_text = f"{timings['wall_s']:.2f}s" if timings["wall_s"] != "" else eval_status
                    print(
                        f"  progress: new_configs={new_configs}/{total_remaining} "
                        f"last={uid} mode={mode} p={p_val} K={k} N={n_requested} "
                        f"wall={wall_text} cfg_elapsed={elapsed_config:.2f}s "
                        f"elapsed={elapsed_total/60:.1f}min"
                    )
            if args.max_configs is not None and new_configs >= args.max_configs:
                break
        if args.max_configs is not None and new_configs >= args.max_configs:
            break
        print(f"  {uid}: added {new_configs - protein_new_start} assisted configs this run")

    progress.close()

    all_rows = _dedupe(existing_rows + new_rows)
    _write_rows(out_path, all_rows, FIELDNAMES)
    _write_rows(completed_path, [dict(zip(COMPLETION_FIELDS, _completion_key(r))) for r in all_rows], list(COMPLETION_FIELDS))
    _write_rows(summary_path, [{
        "data_dir": str(data),
        "benchmark_fasta": str(Path(args.benchmark_fasta).resolve()) if args.benchmark_fasta else "",
        "output_csv": str(out_path),
        "completed_keys_csv": str(completed_path),
        "proteins_found": len(valid_dirs),
        "proteins_seen_this_run": len(loaded),
        "proteins_scored_total": len({r["protein_id"] for r in all_rows}),
        "rows_total": len(all_rows),
        "rows_reused": len(existing_rows),
        "rows_added": len(new_rows),
        "force_recompute": args.force,
        "skipped_no_query_or_aa": ";".join(skipped_no_ref),
        "short_homolog_messages": len(short_homologs),
        "execution_mode": EXECUTION_MODE,
        "device": torch.cuda.get_device_name(0),
    }])
    print(f"Added {len(new_rows)} rows; total T4 rows now {len(all_rows)}")
    print(f"Wrote {out_path}")
    return 0


def _make_progress(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit="cfg", dynamic_ncols=True)
    except Exception:
        return _TextProgress(total, desc)


class _TextProgress:
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc
        self.n = 0
        print(f"{desc}: 0/{total}")

    def update(self, step: int = 1):
        self.n += step

    def close(self):
        print(f"{self.desc}: {self.n}/{self.total} complete")


def _count_remaining_configs(
    loaded,
    data: Path,
    completed: set[tuple[str, str, str, str, str, str]],
    k_values: list[int],
    p_values: list[int],
    n_values: list[int],
    include_static: bool,
) -> int:
    total = 0
    for uid, q3, _projected_rows in loaded:
        aa_path = data / uid / "query_aa.fasta"
        if q3 is None or not aa_path.exists():
            continue
        for n_requested in n_values:
            if include_static:
                for k in k_values:
                    if _make_key(uid, "static", "", k, n_requested) not in completed:
                        total += 1
            for p in p_values:
                for k in k_values:
                    if _make_key(uid, "prefix", p, k, n_requested) not in completed:
                        total += 1
    return total


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None or str(value).strip() == "":
        return list(default)
    if str(value).strip().lower() in {"none", "static-only", "static_only"}:
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _read_id_filter(fasta_path: str | None) -> set[str] | None:
    if fasta_path is None or str(fasta_path).strip() == "":
        return None
    path = Path(fasta_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"benchmark FASTA not found: {path}")
    ids = {name.split()[0].lstrip(">").strip() for name, _seq in read_fasta(path)}
    if not ids:
        raise ValueError(f"benchmark FASTA has no records: {path}")
    return ids


def _count_fasta_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.startswith(">"))


def _empty_timings():
    return {
        "encoder_s": "",
        "drafter_s": "",
        "drafter_forward_calls": "",
        "decoder_s": "",
        "wall_s": "",
        "peak_vram_gb": "",
        "seq": "",
    }


class FoldingProfileAssistant(PreTrainedModel, GenerationMixin):
    config_class = T5Config

    def __init__(self, config, encoder, three_di_token_ids, token_id_to_3di, device):
        super().__init__(config)
        self._encoder = encoder
        self._di_ids = torch.tensor(three_di_token_ids, device=device, dtype=torch.long)
        self._token_id_to_3di = token_id_to_3di
        self._device = device
        self.config.is_encoder_decoder = True
        self.config.decoder_start_token_id = config.decoder_start_token_id
        self.generation_config = GenerationConfig(
            num_assistant_tokens=5,
            num_assistant_tokens_schedule="constant",
            do_sample=False,
            max_length=3000,
        )
        self._active = None
        self._mode = "static"
        self._p = 0
        self.forward_s = 0.0
        self.forward_calls = 0

    def set_active(self, drafter, *, mode: str, p: int):
        self._active = drafter
        self._mode = mode
        self._p = p
        self.forward_s = 0.0
        self.forward_calls = 0

    def get_encoder(self):
        return self._encoder

    def _validate_model_kwargs(self, model_kwargs):
        return

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor, model_kwargs, model_input_name, generation_config
    ):
        # The HMM/profile assistant does not encode the AA prompt. HuggingFace's
        # default encoder-decoder assistant path tries to run a T5 encoder over
        # decoder token ids, which can trigger CUDA index errors. A tiny dummy
        # encoder output is enough because forward() ignores encoder_outputs.
        if "encoder_outputs" not in model_kwargs:
            batch = inputs_tensor.shape[0] if inputs_tensor is not None else 1
            hidden = getattr(self.config, "d_model", 1024)
            model_kwargs["encoder_outputs"] = BaseModelOutput(
                last_hidden_state=torch.zeros(batch, 1, hidden, device=self._device, dtype=torch.float16)
            )
        return model_kwargs

    def prepare_inputs_for_generation(self, decoder_input_ids, encoder_outputs=None, **kwargs):
        return {"decoder_input_ids": decoder_input_ids, "encoder_outputs": encoder_outputs}

    def forward(self, decoder_input_ids=None, encoder_outputs=None, **kwargs):
        t0 = time.perf_counter()
        seq_len = decoder_input_ids.shape[1]
        vocab = self.config.vocab_size
        logits = torch.full((1, seq_len, vocab), -1e4, device=decoder_input_ids.device)
        if self._active is not None:
            decoded = decoder_input_ids[0, 1:].tolist()
            for pos in range(seq_len):
                if self._mode == "prefix":
                    prefix = [self._token_id_to_3di[t] for t in decoded[:pos] if t in self._token_id_to_3di]
                    row = _prefix_log_probs(self._active, prefix, self._p)
                else:
                    row = _static_log_probs(self._active, pos)
                row_t = torch.from_numpy(row).to(self._device, dtype=logits.dtype)
                logits[0, pos, self._di_ids] = row_t
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self.forward_s += time.perf_counter() - t0
        self.forward_calls += 1
        return Seq2SeqLMOutput(logits=logits)


def run_generation(model, tokenizer, aa: str, device, assistant: FoldingProfileAssistant | None):
    _clear_cuda()
    encoded = tokenizer([_format_aa(aa)], add_special_tokens=True, return_tensors="pt").to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    with torch.inference_mode():
        t0 = time.perf_counter()
        encoder_outputs = model.get_encoder()(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)
        torch.cuda.synchronize()
        encoder_s = time.perf_counter() - t0

        if assistant is not None:
            assistant.forward_s = 0.0
            assistant.forward_calls = 0
        gen_kwargs = dict(
            attention_mask=encoded.attention_mask,
            encoder_outputs=encoder_outputs,
            max_length=len(aa) + 2,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
        if assistant is not None:
            gen_kwargs["assistant_model"] = assistant

        t0 = time.perf_counter()
        output = model.generate(**gen_kwargs)
        torch.cuda.synchronize()
        decoder_total_s = time.perf_counter() - t0
        drafter_s = assistant.forward_s if assistant is not None else 0.0
        wall_s = encoder_s + decoder_total_s
        seq = _decode_3di(tokenizer, output[0])[:len(aa)]
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

    del encoded, encoder_outputs, output
    gc.collect()
    _clear_cuda()
    return {
        "encoder_s": encoder_s,
        "drafter_s": drafter_s,
        "drafter_forward_calls": assistant.forward_calls if assistant is not None else 0,
        "decoder_s": max(decoder_total_s - drafter_s, 0.0),
        "wall_s": wall_s,
        "peak_vram_gb": peak_vram_gb,
        "seq": seq,
    }


def _match_mask(drafter, ref, ref_idx, mode, p_val):
    if mode == "static":
        match = [bool(drafter.static_argmax[j] == ref_idx[j] and ref_idx[j] >= 0) for j in range(len(ref))]
    else:
        p = int(p_val)
        match = [bool(drafter.argmax_for_prefix(ref[:j], p) == ref_idx[j] and ref_idx[j] >= 0) for j in range(len(ref))]
    return match, float(np.mean(match))


def _static_log_probs(drafter, pos: int) -> np.ndarray:
    pos = min(pos, drafter.L - 1)
    counts = drafter.static_counts[pos]
    return np.log(counts / counts.sum()).astype(np.float32)


def _prefix_log_probs(drafter, prefix_syms: list[str], max_p: int) -> np.ndarray:
    pos = len(prefix_syms)
    if pos >= drafter.L:
        counts = np.ones(drafter.S, dtype=np.float64)
        return np.log(counts / counts.sum()).astype(np.float32)
    for ctx_len in range(min(max_p, pos), -1, -1):
        ctx = tuple(prefix_syms[-ctx_len:]) if ctx_len > 0 else tuple()
        key = (pos, ctx)
        if key in drafter.context_counts:
            counts = drafter.context_counts[key] + drafter.smoothing
            return np.log(counts / counts.sum()).astype(np.float32)
    return _static_log_probs(drafter, pos)


def _row(*, uid, protein_i, length, pipeline, mode, p, k, n_requested, n_used, n_avail,
         timings, seq, ref, mean_k, acceptance_rate, mean_accepted, n_steps,
         drafter_accuracy, speedup_runtime, drafter_build_s=0.0,
         eval_status="ok", error_message=""):
    drafter_predict_s = timings["drafter_s"]
    drafter_total_s = "" if drafter_predict_s == "" else drafter_build_s + drafter_predict_s
    return {
        "protein_id": uid,
        "protein_index": protein_i,
        "length": length,
        "pipeline": pipeline,
        "mode": mode,
        "p": p,
        "K": k,
        "n_homologs_requested": n_requested,
        "n_homologs_used": n_used,
        "n_homologs_avail": n_avail,
        "encoder_s": _r(timings["encoder_s"]),
        "drafter_build_s": _r(drafter_build_s),
        "drafter_predict_s": _r(drafter_predict_s),
        "drafter_s": _r(drafter_total_s),
        "drafter_forward_calls": timings["drafter_forward_calls"],
        "decoder_s": _r(timings["decoder_s"]),
        "wall_s": _r(timings["wall_s"]),
        "peak_vram_gb": _r(timings["peak_vram_gb"]),
        "speedup_runtime": _r(speedup_runtime) if speedup_runtime != "" else "",
        "speedup_theoretical_mean_k": _r(mean_k),
        "acceptance_rate": _r(acceptance_rate),
        "mean_accepted": _r(mean_accepted),
        "mean_tokens_per_step": _r(mean_k),
        "n_steps": n_steps,
        "drafter_accuracy": _r(drafter_accuracy) if drafter_accuracy != "" else "",
        "exact_match": seq == ref,
        "execution_mode": EXECUTION_MODE,
        "eval_status": eval_status,
        "error_message": error_message,
    }


FIELDNAMES = [
    "protein_id", "protein_index", "length", "pipeline", "mode", "p", "K",
    "n_homologs_requested", "n_homologs_used", "n_homologs_avail",
    "encoder_s", "drafter_build_s", "drafter_predict_s", "drafter_s",
    "drafter_forward_calls", "decoder_s", "wall_s", "peak_vram_gb",
    "speedup_runtime", "speedup_theoretical_mean_k", "acceptance_rate",
    "mean_accepted", "mean_tokens_per_step", "n_steps", "drafter_accuracy",
    "exact_match", "execution_mode", "eval_status", "error_message",
]


def _build_alphabet(loaded):
    alpha = set()
    for _, _, projected_rows in loaded:
        for row in projected_rows:
            alpha |= {c for c in row if c not in GAP_CHARS}
    return {c: i for i, c in enumerate(sorted(alpha))}


def _format_aa(seq: str) -> str:
    return "<AA2fold> " + " ".join(list(seq.upper()))


def _decode_3di(tokenizer, token_ids) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return "".join(text.split()).replace("<AA2fold>", "").lower()


def _clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _r(value):
    if value == "":
        return ""
    return round(float(value), 6)


def _short_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:500]


def _make_key(protein_id, mode, p, k, n_requested):
    return (str(protein_id), str(mode), _norm(p), _norm(k), _norm(n_requested), EXECUTION_MODE)


def _completion_key(row):
    return (
        str(row["protein_id"]),
        str(row["mode"]),
        _norm(row.get("p", "")),
        _norm(row.get("K", "")),
        _norm(row.get("n_homologs_requested", "")),
        str(row.get("execution_mode", EXECUTION_MODE) or EXECUTION_MODE),
    )


def _has_key(row):
    return all(field in row for field in COMPLETION_FIELDS)


def _norm(value):
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    try:
        f = float(text)
    except ValueError:
        return text
    return str(int(f)) if f.is_integer() else text


def _existing_baseline_wall(rows, uid):
    for row in rows:
        if row.get("protein_id") == uid and row.get("mode") == "benchmark" and row.get("wall_s"):
            return float(row["wall_s"])
    return None


def _dedupe(rows):
    by_key = {}
    for row in rows:
        if _has_key(row):
            by_key[_completion_key(row)] = _normalize_row(row)
    return [by_key[key] for key in sorted(by_key)]


def _flush_checkpoints(out_path: Path, completed_path: Path, existing_rows, new_rows):
    all_rows = _dedupe(existing_rows + new_rows)
    _write_rows(out_path, all_rows, FIELDNAMES)
    _write_rows(
        completed_path,
        [dict(zip(COMPLETION_FIELDS, _completion_key(r))) for r in all_rows],
        list(COMPLETION_FIELDS),
    )


def _read_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _normalize_row(row):
    normalized = dict(row)
    for field in FIELDNAMES:
        normalized.setdefault(field, "")
    if normalized.get("eval_status", "") == "":
        normalized["eval_status"] = "ok"
    normalized.setdefault("error_message", "")
    return normalized


def _write_rows(path: Path, rows, fieldnames=None):
    if not rows:
        path.write_text("")
        return
    keys = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([_normalize_row(row) for row in rows])


if __name__ == "__main__":
    raise SystemExit(main())
