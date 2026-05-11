"""H1 of hmm_plan.md — build + validate the PF00535 profile HMM.

Steps:
  1. Download the Pfam PF00535 SEED alignment from InterPro (Stockholm).
     (The plan said "Full". Two deviations:
        - Full alignments don't ship with #=GC RF (HMMER reference annotation).
        - InterPro's SEED dump also strips the match/insert convention:
          every gap becomes '.', every residue becomes upper-case. So we
          can't reconstruct RF from the alignment text either.
      Either way, Pfam itself builds the family HMM from SEED, not Full --
      and with architecture='fast' (symfrac-based match-column selection)
      pyhmmer reproduces Pfam's distributed HMM bit-for-bit. We verified
      this once by downloading Pfam's official PF00535 HMM: same M=168,
      same consensus, same per-residue log-likelihoods on P39621.)
  2. Inspect the MSA (# seqs, avg length, gap fraction).
  3. Build a profile HMM with pyhmmer.plan7.Builder (architecture='fast').
  4. Print the consensus and check for the GT-2 DXD motif.
  5. Align the HMM to the in-family target (P39621, SpsA) with hmmalign.
  6. Length-prune to E[L_target x 20] with one row per target residue.
  7. Validation 1: argmax(E) vs true AA -> sequence identity.
  8. Validation 2: sum_j log E[j, true_aa(j)] in-family >> out-of-family.

Run from prostT5/:
    python build_hmm.py

Artifacts (gitignored) go to prostT5/hmm_data/:
    PF00535_seed.sto    Pfam SEED alignment
    PF00535.hmm         pyhmmer-built HMM
    in_family.fasta     P39621 AA (UniProt)
    out_family.fasta    P04637 AA (UniProt)
"""

from __future__ import annotations

import gzip
import math
import sys
import urllib.request
from pathlib import Path

try:
    import pyhmmer
except ImportError:
    sys.exit(
        "pyhmmer is not installed. Install it with:\n"
        "    pip install pyhmmer\n"
        "(on Colab/Linux a manylinux wheel is available; on Mac it needs CC=clang)."
    )


HERE = Path(__file__).resolve().parent
HMM_DATA = HERE / "hmm_data"
HMM_DATA.mkdir(exist_ok=True)

PFAM_ID = "PF00535"
PFAM_URL = (
    f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{PFAM_ID}/"
    "?annotation=alignment:seed"
)

MSA_PATH = HMM_DATA / f"{PFAM_ID}_seed.sto"
HMM_PATH = HMM_DATA / f"{PFAM_ID}.hmm"

# Validation targets. P39621 = SpsA (B. subtilis), the in-family GT-2 added to
# TEST_IDS in the plan; P04637 = human p53, similar length, not a GT.
IN_FAMILY_UID = "P39621"
OUT_FAMILY_UID = "P04637"

# Canonical order used by the ProstT5 AA head (alphabetical). pyhmmer's amino
# alphabet uses the same ordering for its first 20 canonical symbols, so
# alphabet.symbols.index(c) == AA_ORDER.index(c) for c in AA_ORDER -- we still
# look up via alphabet.symbols to stay format-agnostic.
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def fetch_uniprot_aa(uid: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    with urllib.request.urlopen(url, timeout=30) as r:
        body = r.read().decode("utf-8")
    seq = "".join(
        line.strip() for line in body.splitlines() if not line.startswith(">")
    )
    if not seq:
        raise RuntimeError(f"empty UniProt FASTA for {uid}")
    return seq


def fetch_pfam_msa(dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  cached: {dst}")
        return
    print(f"  downloading {PFAM_ID} SEED alignment from {PFAM_URL}")
    with urllib.request.urlopen(PFAM_URL, timeout=120) as r:
        data = r.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if not data.lstrip().startswith(b"# STOCKHOLM"):
        raise RuntimeError(
            "downloaded payload does not look like Stockholm "
            f"(starts with {data[:60]!r})"
        )
    dst.write_bytes(data)
    print(f"  wrote {dst} ({dst.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# MSA inspection
# ---------------------------------------------------------------------------

def _as_text_seq(s) -> str:
    seq = s.sequence
    if isinstance(seq, (bytes, bytearray)):
        seq = seq.decode("ascii", errors="replace")
    return seq


def inspect_msa(msa) -> None:
    aln = msa.alignment
    n_seqs = len(aln)
    aln_len = len(aln[0]) if n_seqs else 0
    K = msa.alphabet.K
    gap_code = K  # 20 for amino: position of '-' in alphabet.symbols
    raw_lens = [sum(1 for c in row if c < K) for row in aln]
    avg_len = sum(raw_lens) / max(n_seqs, 1)
    total = n_seqs * aln_len if aln_len else 1
    gaps = sum(sum(1 for c in row if c == gap_code) for row in aln)
    gap_frac = gaps / total

    rf = getattr(msa, "reference", None)
    if isinstance(rf, (bytes, bytearray)):
        rf = rf.decode("ascii", errors="replace")
    if rf:
        n_match = sum(1 for c in rf if c not in (".", "-"))
        rf_msg = f"{n_match} (from #=GC RF)"
    else:
        rf_msg = "(no #=GC RF annotation)"

    print(f"  sequences:       {n_seqs}")
    print(f"  alignment len:   {aln_len}")
    print(f"  avg seq len:     {avg_len:.1f}  (min={min(raw_lens)}, max={max(raw_lens)})")
    print(f"  gap fraction:    {gap_frac:.3f}")
    print(f"  match columns:   {rf_msg}")


# ---------------------------------------------------------------------------
# HMM build
# ---------------------------------------------------------------------------

def build_hmm(msa_path: Path, hmm_path: Path, alphabet) -> "pyhmmer.plan7.HMM":
    with pyhmmer.easel.MSAFile(
        str(msa_path), format="stockholm", digital=True, alphabet=alphabet
    ) as f:
        msa = f.read()
    if msa is None:
        raise RuntimeError(f"no MSA found in {msa_path}")
    if not msa.name:
        msa.name = PFAM_ID.encode()

    print(f"\n[MSA inspection — {msa_path.name}]")
    inspect_msa(msa)

    # architecture='fast' picks match columns by symfrac (default 0.5). This is
    # what Pfam itself uses for PF00535: their distributed HMM has M=168 and our
    # build with these defaults reproduces it bit-for-bit.
    builder = pyhmmer.plan7.Builder(alphabet, architecture="fast", seed=42)
    background = pyhmmer.plan7.Background(alphabet)
    hmm, _, _ = builder.build_msa(msa, background)

    with open(hmm_path, "wb") as f:
        hmm.write(f)

    consensus = hmm.consensus
    print(f"\n[HMM built -> {hmm_path.name}]")
    print(f"  M (match columns): {hmm.M}")
    print(f"  consensus ({len(consensus)} chars):")
    for i in range(0, len(consensus), 80):
        print(f"    {consensus[i:i+80]}")

    # GT-2 DXD-motif smoke test: two Asps separated by any AA.
    upper = consensus.upper()
    dxd_hits = [
        i + 1
        for i in range(len(upper) - 2)
        if upper[i] == "D" and upper[i + 2] == "D"
    ]
    if dxd_hits:
        print(f"  DXD motif in consensus: YES (match cols {dxd_hits[:5]}"
              f"{'...' if len(dxd_hits) > 5 else ''})")
    else:
        print("  DXD motif in consensus: NO — investigate, GT-2 family is "
              "defined by this motif.")
    return hmm


# ---------------------------------------------------------------------------
# Length-pruned emission matrix
# ---------------------------------------------------------------------------

def emission_matrix(
    hmm, alphabet, aa_seq: str, uid: str
) -> tuple[list[list[float]], list[int]]:
    """Align `aa_seq` to `hmm` and return (E, match_col).

    E has shape (L_target, 20) -- one row per target residue, columns in
    AA_ORDER. match_col[j] is the 1-based HMM match column the j-th target
    residue aligned to, or 0 if it landed in an insert state (we use the HMM
    background distribution for those rows).

    Reads the per-column aligned row from `aln.alignment[i]` (a str) -- the
    TextSequence.sequence accessor returns the ungapped residue string, which
    is not what we need. RF (`aln.reference`) tells us which alignment columns
    are match vs insert.
    """
    seq = pyhmmer.easel.TextSequence(
        name=uid.encode(), sequence=aa_seq
    ).digitize(alphabet)
    # trim=False keeps the full target in the alignment (we need one row per
    # target residue, including residues outside the matched region).
    aln = pyhmmer.hmmer.hmmalign(hmm, [seq], trim=False, all_consensus_cols=True)
    aligned = aln.alignment[0]
    rf = aln.reference
    if rf is None or len(rf) != len(aligned):
        raise RuntimeError(
            f"hmmalign output has no usable RF (rf={rf!r}, aligned len={len(aligned)})"
        )

    me = hmm.match_emissions  # (M+1, Kp); rows 1..M are the match states
    bg = pyhmmer.plan7.Background(alphabet).residue_frequencies
    sym_idx = {c: alphabet.symbols.index(c) for c in AA_ORDER}
    bg_row = [float(bg[sym_idx[a]]) for a in AA_ORDER]

    rows: list[list[float]] = []
    match_col: list[int] = []
    m_col = 0  # 1-based counter; increments on every match-column position

    for ch, rf_ch in zip(aligned, rf):
        is_match_col = rf_ch not in (".", "-")
        if is_match_col:
            m_col += 1
        is_residue = ch.isalpha()
        if not is_residue:
            continue  # gap (deletion or pad); no target residue here
        if is_match_col:
            rows.append([float(me[m_col][sym_idx[a]]) for a in AA_ORDER])
            match_col.append(m_col)
        else:
            rows.append(bg_row)
            match_col.append(0)

    if len(rows) != len(aa_seq):
        raise RuntimeError(
            f"length mismatch after hmmalign for {uid}: "
            f"got {len(rows)} rows, expected {len(aa_seq)} "
            f"(aligned width={len(aligned)})"
        )
    bad = [j for j, r in enumerate(rows) if not all(0.0 <= p <= 1.0 + 1e-6 for p in r)]
    if bad:
        raise RuntimeError(f"E rows out of [0,1] at indices {bad[:5]}")
    return rows, match_col


# ---------------------------------------------------------------------------
# Validations
# ---------------------------------------------------------------------------

def validation_1(
    E: list[list[float]], match_col: list[int], aa_seq: str, uid: str
) -> tuple[float, float]:
    pred = "".join(
        AA_ORDER[max(range(20), key=lambda k: row[k])] for row in E
    )
    n_total = len(aa_seq)
    n_match = sum(1 for c in match_col if c > 0)
    n_ins = n_total - n_match
    hits_total = sum(1 for a, b in zip(pred, aa_seq) if a == b)
    hits_match = sum(
        1 for j, (a, b) in enumerate(zip(pred, aa_seq))
        if match_col[j] > 0 and a == b
    )
    overall = hits_total / n_total
    match_only = hits_match / n_match if n_match else 0.0
    print(f"\n[Validation 1 — argmax(E) vs true AA, {uid}]")
    print(f"  L={n_total}  match-col residues={n_match}  insert-col residues={n_ins}")
    print(f"  overall identity:           {overall*100:5.1f}%  ({hits_total}/{n_total})")
    print(f"  match-column identity only: {match_only*100:5.1f}%  ({hits_match}/{n_match})")
    # The threshold in hmm_plan.md is on the meaningful subset -- insert-state
    # rows use the background distribution (its argmax is just the most common
    # AA), so include them dilutes the signal. Gate on match-only identity.
    if match_only < 0.20:
        print("  *** FAIL: match-only < 20% — alignment is broken, debug")
    elif match_only < 0.30:
        print("  ** borderline (match-only < 30%)")
    else:
        print("  OK (match-only >= 30%)")
    return overall, match_only


def total_logp(E, aa_seq) -> tuple[float, float]:
    s = 0.0
    for row, aa in zip(E, aa_seq):
        idx = AA_ORDER.find(aa)
        if idx < 0:
            p = 1e-12  # non-canonical residue (e.g. X)
        else:
            p = row[idx] if row[idx] > 0 else 1e-12
        s += math.log(p)
    return s, s / len(aa_seq)


def validation_2(
    E_in, aa_in, uid_in, E_out, aa_out, uid_out
) -> None:
    in_tot, in_avg = total_logp(E_in, aa_in)
    out_tot, out_avg = total_logp(E_out, aa_out)
    print("\n[Validation 2 — log-likelihood of true AA under E]")
    print(f"  {uid_in:>10s} (in-family,  L={len(aa_in):4d}):  "
          f"total={in_tot:9.1f}  per-residue={in_avg:+.3f}")
    print(f"  {uid_out:>10s} (out-of-family, L={len(aa_out):4d}):  "
          f"total={out_tot:9.1f}  per-residue={out_avg:+.3f}")
    print(f"  per-residue delta = {in_avg - out_avg:+.3f}  "
          "(expect in-family >> out-of-family)")
    if in_avg <= out_avg:
        print("  *** FAIL: in-family does not dominate -- HMM or alignment is suspect")
    else:
        print("  OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    alphabet = pyhmmer.easel.Alphabet.amino()

    print("=== Step 1: download Pfam PF00535 SEED alignment ===")
    fetch_pfam_msa(MSA_PATH)

    print("\n=== Step 2: build HMM ===")
    hmm = build_hmm(MSA_PATH, HMM_PATH, alphabet)

    print("\n=== Step 3: fetch validation targets from UniProt ===")
    in_aa = fetch_uniprot_aa(IN_FAMILY_UID)
    out_aa = fetch_uniprot_aa(OUT_FAMILY_UID)
    (HMM_DATA / "in_family.fasta").write_text(f">{IN_FAMILY_UID}\n{in_aa}\n")
    (HMM_DATA / "out_family.fasta").write_text(f">{OUT_FAMILY_UID}\n{out_aa}\n")
    print(f"  {IN_FAMILY_UID}:  L={len(in_aa)}")
    print(f"  {OUT_FAMILY_UID}: L={len(out_aa)}")
    # Sanity: does the SpsA sequence literally contain a DXD motif?
    dxd_in_seq = [
        i + 1 for i in range(len(in_aa) - 2)
        if in_aa[i] == "D" and in_aa[i + 2] == "D"
    ]
    print(f"  D.D positions in {IN_FAMILY_UID}: {dxd_in_seq}")

    print("\n=== Step 4: hmmalign + length-prune ===")
    E_in, mcol_in = emission_matrix(hmm, alphabet, in_aa, IN_FAMILY_UID)
    E_out, mcol_out = emission_matrix(hmm, alphabet, out_aa, OUT_FAMILY_UID)
    print(f"  {IN_FAMILY_UID}:  {sum(c > 0 for c in mcol_in)}/{len(in_aa)} "
          "residues at match columns")
    print(f"  {OUT_FAMILY_UID}: {sum(c > 0 for c in mcol_out)}/{len(out_aa)} "
          "residues at match columns")

    print("\n=== Step 5: validations ===")
    validation_1(E_in, mcol_in, in_aa, IN_FAMILY_UID)
    validation_2(E_in, in_aa, IN_FAMILY_UID, E_out, out_aa, OUT_FAMILY_UID)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
