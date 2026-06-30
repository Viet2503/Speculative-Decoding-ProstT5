"""Offline per-protein evaluation table for the folding HMM/profile drafter.

This uses only prostT5/prostT5_benchmarks/folding_MSA. It
records the metrics we can measure without running ProstT5 generation:

* HMM drafter build time
* HMM drafter prediction time
* acceptance rate
* theoretical speedup / mean k

The true encoder time, decoder time, and peak CUDA vRAM require the real
ProstT5 CUDA generation path, so those columns are included but left empty when
this script runs in offline mode.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from folding_profile_sweep import (
    FoldingProfileDrafter,
    GAP_CHARS,
    K_VALUES,
    N_HOMOLOG_VALUES,
    P_VALUES,
    load_protein,
    simulate_fixed_k,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "prostT5_benchmarks" / "folding_MSA"
DEFAULT_OUT = SCRIPT_DIR / "results" / "folding_hmm_results" / "eval_result"
EXECUTION_MODE = "offline_cpu_hmm_profile"
COMPLETION_KEY_FIELDS = (
    "protein_id",
    "mode",
    "p",
    "K",
    "n_homologs_requested",
    "execution_mode",
)


def _build_alphabet(loaded: list[tuple[str, str | None, list[str]]]) -> dict[str, int]:
    alpha = set()
    for _, _, projected_rows in loaded:
        for row in projected_rows:
            alpha |= {c for c in row if c not in GAP_CHARS}
    return {c: i for i, c in enumerate(sorted(alpha))}


def _mean_or_nan(values: list[float]) -> float:
    return round(float(np.mean(values)), 6) if values else math.nan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="path to folding_MSA")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--filename", default="folding_hmm_eval_result.csv")
    parser.add_argument("--force", action="store_true",
                        help="recompute all rows instead of resuming from the existing CSV")
    args = parser.parse_args()

    data = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename
    summary_path = out_dir / "folding_hmm_eval_run_summary.csv"
    completed_path = out_dir / "folding_hmm_eval_completed_keys.csv"

    existing_rows = [] if args.force else _read_existing_rows(out_path)
    completed = {_completion_key(r) for r in existing_rows if _has_completion_key(r)}
    if existing_rows:
        print(f"Resume: loaded {len(existing_rows)} existing rows from {out_path}")
        print(f"Resume: {len(completed)} completed configuration keys")
    elif args.force:
        print("Force recompute enabled: ignoring existing result CSV.")

    dirs = sorted(p for p in data.iterdir() if p.is_dir())
    valid_dirs = [p for p in dirs if (p / "homologs_projected_to_query_3di.fasta").exists()]
    loaded = [load_protein(d) for d in valid_dirs]
    sym_to_idx = _build_alphabet(loaded)

    rows: list[dict] = []
    skipped_existing = 0
    skipped_no_ref: list[str] = []
    short_homolog_messages: list[str] = []

    for protein_i, (uid, q3, projected_rows) in enumerate(loaded, start=1):
        if q3 is None:
            skipped_no_ref.append(uid)
            continue

        benchmark_key = _key(uid, "benchmark", "", "", "")
        if benchmark_key in completed:
            skipped_existing += 1
        else:
            rows.append(_benchmark_row(
                uid=uid,
                protein_index=protein_i,
                length=len(q3),
                homologs_avail=len(projected_rows),
            ))

        ref = list(q3)
        ref_idx = np.array([sym_to_idx.get(c, -1) for c in ref])
        length = len(ref)
        homologs_avail = len(projected_rows)

        for n_requested in N_HOMOLOG_VALUES:
            keys_for_n = [_key(uid, "static", "", k, n_requested) for k in K_VALUES]
            keys_for_n.extend(
                _key(uid, "prefix", p, k, n_requested)
                for p in P_VALUES
                for k in K_VALUES
            )
            missing_for_n = [key for key in keys_for_n if key not in completed]
            if not missing_for_n:
                skipped_existing += len(keys_for_n)
                continue

            rows_n = projected_rows[:n_requested]
            n_used = len(rows_n)
            if homologs_avail < n_requested:
                msg = f'protein "{uid}" has only {homologs_avail} homologs, requires {n_requested}; using {n_used}.'
                print(msg)
                short_homolog_messages.append(msg)

            t0 = time.perf_counter()
            drafter = FoldingProfileDrafter(rows_n, length, sym_to_idx)
            drafter_build_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            match_static = [
                bool(drafter.static_argmax[j] == ref_idx[j] and ref_idx[j] >= 0)
                for j in range(length)
            ]
            static_predict_s = time.perf_counter() - t0
            static_acc = float(np.mean(match_static))

            for k in K_VALUES:
                key = _key(uid, "static", "", k, n_requested)
                if key in completed:
                    skipped_existing += 1
                    continue
                steps, mean_accepted, mean_k, acceptance_rate = simulate_fixed_k(match_static, k)
                rows.append(_row(
                    uid=uid,
                    length=length,
                    protein_index=protein_i,
                    n_requested=n_requested,
                    n_used=n_used,
                    homologs_avail=homologs_avail,
                    mode="static",
                    p="",
                    k=k,
                    drafter_build_s=drafter_build_s,
                    drafter_predict_s=static_predict_s,
                    drafter_accuracy=static_acc,
                    steps=steps,
                    mean_accepted=mean_accepted,
                    mean_k=mean_k,
                    acceptance_rate=acceptance_rate,
                ))

            for p in P_VALUES:
                keys_for_p = [_key(uid, "prefix", p, k, n_requested) for k in K_VALUES]
                missing_for_p = [key for key in keys_for_p if key not in completed]
                if not missing_for_p:
                    skipped_existing += len(keys_for_p)
                    continue

                t0 = time.perf_counter()
                match_p = []
                for j in range(length):
                    pred = drafter.argmax_for_prefix(ref[:j], p)
                    match_p.append(bool(pred == ref_idx[j] and ref_idx[j] >= 0))
                prefix_predict_s = time.perf_counter() - t0
                prefix_acc = float(np.mean(match_p))

                for k in K_VALUES:
                    key = _key(uid, "prefix", p, k, n_requested)
                    if key in completed:
                        skipped_existing += 1
                        continue
                    steps, mean_accepted, mean_k, acceptance_rate = simulate_fixed_k(match_p, k)
                    rows.append(_row(
                        uid=uid,
                        length=length,
                        protein_index=protein_i,
                        n_requested=n_requested,
                        n_used=n_used,
                        homologs_avail=homologs_avail,
                        mode="prefix",
                        p=p,
                        k=k,
                        drafter_build_s=drafter_build_s,
                        drafter_predict_s=prefix_predict_s,
                        drafter_accuracy=prefix_acc,
                        steps=steps,
                        mean_accepted=mean_accepted,
                        mean_k=mean_k,
                        acceptance_rate=acceptance_rate,
                    ))

    all_rows = _dedupe_rows(existing_rows + rows)
    _write_csv(out_path, all_rows, fieldnames=_fieldnames())
    _write_completed_keys(completed_path, all_rows)

    summary_rows = [{
        "data_dir": str(data),
        "output_csv": str(out_path),
        "completed_keys_csv": str(completed_path),
        "proteins_found": len(valid_dirs),
        "proteins_scored": len({r["protein_id"] for r in all_rows}),
        "rows_total": len(all_rows),
        "rows_reused": len(existing_rows),
        "rows_added": len(rows),
        "rows_skipped_existing": skipped_existing,
        "force_recompute": args.force,
        "skipped_no_query_3di": ";".join(skipped_no_ref),
        "short_homolog_messages": len(short_homolog_messages),
        "execution_mode": EXECUTION_MODE,
        "encoder_decoder_vram_note": "encoder_s, decoder_s, wall_s, speedup_runtime, peak_vram_gb require CUDA ProstT5 generation and are not measured by this offline run",
    }]
    _write_csv(summary_path, summary_rows)

    print(f"Added {len(rows)} new rows; reused {len(existing_rows)} existing rows.")
    print(f"Wrote {len(all_rows)} total rows to {out_path}")
    print(f"Wrote completion keys to {completed_path}")
    print(f"Wrote run summary to {summary_path}")
    if skipped_no_ref:
        print("Skipped no query_3di: " + ", ".join(skipped_no_ref))
    return 0


def _row(
    *,
    uid: str,
    length: int,
    protein_index: int,
    n_requested: int,
    n_used: int,
    homologs_avail: int,
    mode: str,
    p: int | str,
    k: int,
    drafter_build_s: float,
    drafter_predict_s: float,
    drafter_accuracy: float,
    steps: int,
    mean_accepted: float,
    mean_k: float,
    acceptance_rate: float,
) -> dict:
    return {
        "protein_id": uid,
        "protein_index": protein_index,
        "length": length,
        "pipeline": "folding_hmm_profile_offline",
        "mode": mode,
        "p": p,
        "K": k,
        "n_homologs_requested": n_requested,
        "n_homologs_used": n_used,
        "n_homologs_avail": homologs_avail,
        "encoder_s": "",
        "drafter_build_s": round(drafter_build_s, 6),
        "drafter_predict_s": round(drafter_predict_s, 6),
        "drafter_s": round(drafter_build_s + drafter_predict_s, 6),
        "decoder_s": "",
        "wall_s": "",
        "peak_vram_gb": "",
        "speedup_runtime": "",
        "speedup_theoretical_mean_k": round(mean_k, 6),
        "acceptance_rate": round(acceptance_rate, 6),
        "mean_accepted": round(mean_accepted, 6),
        "mean_tokens_per_step": round(mean_k, 6),
        "n_steps": steps,
        "drafter_accuracy": round(drafter_accuracy, 6),
        "execution_mode": EXECUTION_MODE,
    }


def _benchmark_row(*, uid: str, protein_index: int, length: int, homologs_avail: int) -> dict:
    return {
        "protein_id": uid,
        "protein_index": protein_index,
        "length": length,
        "pipeline": "prostT5_benchmark",
        "mode": "benchmark",
        "p": "",
        "K": "",
        "n_homologs_requested": "",
        "n_homologs_used": "",
        "n_homologs_avail": homologs_avail,
        "encoder_s": "",
        "drafter_build_s": "",
        "drafter_predict_s": "",
        "drafter_s": "",
        "decoder_s": "",
        "wall_s": "",
        "peak_vram_gb": "",
        "speedup_runtime": "",
        "speedup_theoretical_mean_k": 1.0,
        "acceptance_rate": 1.0,
        "mean_accepted": 0.0,
        "mean_tokens_per_step": 1.0,
        "n_steps": length,
        "drafter_accuracy": "",
        "execution_mode": EXECUTION_MODE,
    }


def _fieldnames() -> list[str]:
    return [
        "protein_id",
        "protein_index",
        "length",
        "pipeline",
        "mode",
        "p",
        "K",
        "n_homologs_requested",
        "n_homologs_used",
        "n_homologs_avail",
        "encoder_s",
        "drafter_build_s",
        "drafter_predict_s",
        "drafter_s",
        "decoder_s",
        "wall_s",
        "peak_vram_gb",
        "speedup_runtime",
        "speedup_theoretical_mean_k",
        "acceptance_rate",
        "mean_accepted",
        "mean_tokens_per_step",
        "n_steps",
        "drafter_accuracy",
        "execution_mode",
    ]


def _key(protein_id: str, mode: str, p: int | str, k: int | str, n_requested: int | str) -> tuple[str, str, str, str, str, str]:
    return (
        str(protein_id),
        str(mode),
        _normalize_p(p),
        _normalize_int_key(k),
        _normalize_int_key(n_requested),
        EXECUTION_MODE,
    )


def _completion_key(row: dict) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["protein_id"]),
        str(row["mode"]),
        _normalize_p(row.get("p", "")),
        _normalize_int_key(row.get("K", "")),
        _normalize_int_key(row.get("n_homologs_requested", "")),
        str(row.get("execution_mode", EXECUTION_MODE) or EXECUTION_MODE),
    )


def _has_completion_key(row: dict) -> bool:
    return all(field in row for field in COMPLETION_KEY_FIELDS)


def _normalize_p(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    try:
        as_float = float(text)
    except ValueError:
        return text
    return str(int(as_float)) if as_float.is_integer() else text


def _normalize_int_key(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return str(int(float(text)))


def _read_existing_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    by_key = {}
    for row in rows:
        if not _has_completion_key(row):
            continue
        by_key[_completion_key(row)] = row
    return [by_key[key] for key in sorted(by_key)]


def _write_completed_keys(path: Path, rows: list[dict]) -> None:
    key_rows = [
        dict(zip(COMPLETION_KEY_FIELDS, _completion_key(row)))
        for row in rows
        if _has_completion_key(row)
    ]
    _write_csv(path, key_rows, fieldnames=list(COMPLETION_KEY_FIELDS))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("")
        return
    keys = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
