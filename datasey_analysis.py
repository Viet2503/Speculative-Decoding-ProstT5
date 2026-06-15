#!/usr/bin/env python3
"""Analyze AA and 3Di FASTA lengths for the ProstT5 benchmark dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "prostT5" / "prostT5_benchmarks" / "benchmark_data"


def parse_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id: str | None = None

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                records[current_id] = ""
            elif current_id is None:
                raise ValueError(f"Sequence found before FASTA header in {path}")
            else:
                records[current_id] += line

    return records


def summarize_lengths(name: str, lengths: list[int]) -> str:
    if not lengths:
        return f"{name}: no records"
    return (
        f"{name}: n={len(lengths)}, "
        f"min={min(lengths)}, "
        f"median={median(lengths):.1f}, "
        f"mean={mean(lengths):.1f}, "
        f"max={max(lengths)}"
    )


def length_stats(lengths: list[int]) -> dict[str, object]:
    if not lengths:
        return {"n": 0, "min": "", "median": "", "mean": "", "max": ""}
    return {
        "n": len(lengths),
        "min": min(lengths),
        "median": f"{median(lengths):.1f}",
        "mean": f"{mean(lengths):.1f}",
        "max": max(lengths),
    }


def build_rows(aa_records: dict[str, str], di_records: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for protein_id in sorted(set(aa_records) | set(di_records)):
        aa_len = len(aa_records[protein_id]) if protein_id in aa_records else None
        di_len = len(di_records[protein_id]) if protein_id in di_records else None

        if aa_len is None:
            status = "missing_AA"
            diff = None
        elif di_len is None:
            status = "missing_3Di"
            diff = None
        elif aa_len == di_len:
            status = "matched"
            diff = 0
        else:
            status = "length_mismatch"
            diff = aa_len - di_len

        rows.append(
            {
                "protein_id": protein_id,
                "aa_length": aa_len if aa_len is not None else "",
                "three_di_length": di_len if di_len is not None else "",
                "aa_minus_3di": diff if diff is not None else "",
                "status": status,
            }
        )
    return rows


def print_table(rows: list[dict[str, object]], limit: int | None) -> None:
    shown = rows if limit is None else rows[:limit]
    headers = ["protein_id", "aa_length", "three_di_length", "aa_minus_3di", "status"]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in shown)) if shown else len(header)
        for header in headers
    }

    print("Per-protein lengths:")
    print("  " + "  ".join(header.ljust(widths[header]) for header in headers))
    print("  " + "  ".join("-" * widths[header] for header in headers))
    for row in shown:
        print("  " + "  ".join(str(row[header]).ljust(widths[header]) for header in headers))

    if limit is not None and len(rows) > limit:
        print(f"  ... {len(rows) - limit} more rows not shown")


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_id", "aa_length", "three_di_length", "aa_minus_3di", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    data_dir: Path,
    aa_path: Path,
    di_path: Path,
    aa_records: dict[str, str],
    di_records: dict[str, str],
    rows: list[dict[str, object]],
) -> None:
    common_ids = sorted(set(aa_records) & set(di_records))
    matched_rows = [row for row in rows if row["status"] == "matched"]
    mismatch_rows = [row for row in rows if row["status"] == "length_mismatch"]
    aa_only = sorted(set(aa_records) - set(di_records))
    di_only = sorted(set(di_records) - set(aa_records))
    aa_stats = length_stats([len(seq) for seq in aa_records.values()])
    di_stats = length_stats([len(seq) for seq in di_records.values()])

    text = f"""# Benchmark Data

This note summarizes the paired AA and 3Di benchmark data used by the ProstT5 folding experiments.

## Location

- Dataset directory: `{data_dir}`
- AA FASTA: `{aa_path}`
- 3Di FASTA: `{di_path}`

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
| AA proteins | {len(aa_records)} |
| 3Di proteins | {len(di_records)} |
| Proteins in both files | {len(common_ids)} |
| Matched AA/3Di lengths | {len(matched_rows)} |
| Length mismatches | {len(mismatch_rows)} |
| Only in AA FASTA | {len(aa_only)} |
| Only in 3Di FASTA | {len(di_only)} |

## Length Statistics

| Sequence type | n | min | median | mean | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| AA | {aa_stats["n"]} | {aa_stats["min"]} | {aa_stats["median"]} | {aa_stats["mean"]} | {aa_stats["max"]} |
| 3Di | {di_stats["n"]} | {di_stats["min"]} | {di_stats["median"]} | {di_stats["mean"]} | {di_stats["max"]} |

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
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing test_set_AA.fasta and test_set_3Di.fasta. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument("--aa-file", default="test_set_AA.fasta", help="AA FASTA filename.")
    parser.add_argument("--three-di-file", default="test_set_3Di.fasta", help="3Di FASTA filename.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the printed per-protein table.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path for per-protein lengths.")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional Markdown report output path.")
    args = parser.parse_args()

    aa_path = args.data_dir / args.aa_file
    di_path = args.data_dir / args.three_di_file

    aa_records = parse_fasta(aa_path)
    di_records = {protein_id: seq.lower() for protein_id, seq in parse_fasta(di_path).items()}
    rows = build_rows(aa_records, di_records)

    common_ids = sorted(set(aa_records) & set(di_records))
    matched_rows = [row for row in rows if row["status"] == "matched"]
    mismatch_rows = [row for row in rows if row["status"] == "length_mismatch"]
    aa_only = sorted(set(aa_records) - set(di_records))
    di_only = sorted(set(di_records) - set(aa_records))

    print(f"Dataset directory: {args.data_dir}")
    print(f"AA FASTA: {aa_path}")
    print(f"3Di FASTA: {di_path}")
    print()
    print(f"AA proteins: {len(aa_records)}")
    print(f"3Di proteins: {len(di_records)}")
    print(f"Proteins in both files: {len(common_ids)}")
    print(f"Matched AA/3Di lengths: {len(matched_rows)}")
    print(f"Length mismatches: {len(mismatch_rows)}")
    print(f"Only in AA FASTA: {len(aa_only)}")
    print(f"Only in 3Di FASTA: {len(di_only)}")
    print()
    print(summarize_lengths("AA lengths, all AA records", [len(seq) for seq in aa_records.values()]))
    print(summarize_lengths("3Di lengths, all 3Di records", [len(seq) for seq in di_records.values()]))
    print(summarize_lengths("AA lengths, common IDs", [len(aa_records[protein_id]) for protein_id in common_ids]))
    print(summarize_lengths("3Di lengths, common IDs", [len(di_records[protein_id]) for protein_id in common_ids]))
    print()

    if mismatch_rows:
        print("Mismatched proteins:")
        for row in mismatch_rows:
            print(
                f"  {row['protein_id']}: "
                f"AA={row['aa_length']} 3Di={row['three_di_length']} "
                f"diff={row['aa_minus_3di']}"
            )
        print()

    print_table(rows, args.limit)

    if args.csv is not None:
        write_csv(rows, args.csv)
        print()
        print(f"Wrote CSV: {args.csv}")

    if args.markdown is not None:
        write_markdown(args.markdown, args.data_dir, aa_path, di_path, aa_records, di_records, rows)
        print()
        print(f"Wrote Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
