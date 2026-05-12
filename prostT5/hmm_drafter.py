"""H2 of hmm_plan.md — HMMDrafter + custom Leviathan greedy verify loop.

Two pieces:
  * `HMMDrafter`: profile-HMM drafter, length-pruned to a target AA sequence.
    The HMM is loaded + aligned once at construction time; `propose(K)` walks
    a cached vocab-id table; `commit(n)` advances the cursor. No model state.
  * `spec_decode_greedy`: Leviathan-style spec decoding for the enc-dec model.
    Encode the 3Di prompt once, maintain the decoder self-attn KV cache, feed
    `[last_token, *proposals]` each step, accept the matching argmax prefix,
    emit the verifier's argmax at the first disagreement (or as the free
    "+1" on full accept), prune the cache to the verified length.

Bit-exact assertions live in `assert_bit_exact()` — see hard constraint #1 in
CLAUDE.md ("speculative decoding must produce token-for-token identical output
to running the full ProstT5 enc-dec under greedy decoding").

H1's `emission_matrix()` (which we know is correct — see `build_hmm.py` and the
H1 validation block of hmm_plan.md) is reused as-is; this file only adds the
H2 surface.

Run a self-contained smoke test (no ProstT5, no GPU needed):
    cd prostT5/
    python hmm_drafter.py --smoke
The smoke test covers the HMMDrafter mechanics + the round-trip 20→vocab map.
The bit-exact end-to-end test needs ProstT5 + a GPU, so it lives as a function
to be called from the notebook (see assert_bit_exact()).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

try:
    import pyhmmer
except ImportError:
    sys.exit(
        "pyhmmer is not installed. Install it with `pip install pyhmmer` "
        "(on Colab/Linux a manylinux wheel is available)."
    )

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from build_hmm import AA_ORDER, emission_matrix  # noqa: E402


# ---------------------------------------------------------------------------
# Tokenizer-identity assertion (hard constraint #2)
# ---------------------------------------------------------------------------

def tokenizer_vocab_hash(tokenizer) -> str:
    """SHA-256 of canonical (token, id) pairs. Two tokenizers with the same
    hash are interchangeable for our purposes (same vocab, same ids)."""
    vocab = tokenizer.get_vocab()  # dict: token -> id
    blob = "\n".join(f"{tok}\t{tid}" for tok, tid in sorted(vocab.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def aa_to_vocab_map(tokenizer) -> torch.LongTensor:
    """Build the 20-class → ProstT5-vocab LongTensor[20] map.

    For each AA letter in AA_ORDER (the canonical alphabetical order used by
    `build_hmm.py` and the AA-CNN head), return the tokenizer's vocab id. We
    try the bare letter first ("A") then the sentencepiece-prefixed form
    ("▁A") so this works for either ProstT5 tokenizer flavor.
    """
    unk = tokenizer.unk_token_id
    ids: list[int] = []
    for c in AA_ORDER:
        tid = tokenizer.convert_tokens_to_ids(c)
        if tid is None or tid == unk:
            tid = tokenizer.convert_tokens_to_ids(f"▁{c}")
        if tid is None or tid == unk:
            raise RuntimeError(
                f"AA letter {c!r} has no vocab entry in this tokenizer "
                f"(both {c!r} and '▁{c}' map to UNK)."
            )
        ids.append(int(tid))
    return torch.tensor(ids, dtype=torch.long)


def token_id_to_aa(tokenizer, token_id: int) -> str:
    """Decode one tokenizer id to a single AA letter, or "" if it is not an
    amino-acid token (for example EOS).
    """
    s = tokenizer.decode([token_id], skip_special_tokens=True)
    s = "".join(s.split())
    return s if len(s) == 1 else ""


def aa_token_ids(tokenizer, token_ids: list[int]) -> list[int]:
    """Filter a token-id list down to single-letter amino-acid tokens."""
    out: list[int] = []
    for token_id in token_ids:
        if token_id_to_aa(tokenizer, token_id):
            out.append(int(token_id))
    return out


# ---------------------------------------------------------------------------
# HMM drafter
# ---------------------------------------------------------------------------

def _load_hmm(hmm_path: str | Path):
    with pyhmmer.plan7.HMMFile(str(hmm_path)) as hf:
        return next(iter(hf))


class HMMDrafter:
    """Profile-HMM drafter, length-pruned to a single target AA sequence.

    Constructor aligns `target_aa_seq` to the HMM (via pyhmmer.hmmer.hmmalign),
    pulls the (L, 20) emission matrix using H1's `emission_matrix`, takes the
    per-row argmax, and maps to ProstT5 vocab ids. After that, `propose(K)` is
    O(K) (a slice) and `commit(n)` is O(1).

    The drafter is **prefix-blind**: rows are indexed by output position only,
    not by what the verifier has actually emitted so far. That's fine for
    bit-exact correctness (the verifier can always reject a bad token), and
    it's what the H1/H2 plan calls for. H5 is the prefix-aware variant.
    """

    def __init__(self, hmm_path: str | Path, target_aa_seq: str, tokenizer):
        # Hard constraint #2: drafter and verifier must share vocab. Snapshot
        # the hash now; the verify loop checks it against the verifier's
        # tokenizer.
        self.tokenizer_hash = tokenizer_vocab_hash(tokenizer)

        alphabet = pyhmmer.easel.Alphabet.amino()
        self._alphabet = alphabet
        self.hmm = _load_hmm(hmm_path)
        self._tokenizer = tokenizer

        e_rows, match_col = emission_matrix(
            self.hmm, alphabet, target_aa_seq, "drafter"
        )
        E = torch.tensor(e_rows, dtype=torch.float32)
        if E.shape != (len(target_aa_seq), 20):
            raise RuntimeError(
                f"emission matrix has unexpected shape {tuple(E.shape)}; "
                f"expected ({len(target_aa_seq)}, 20)"
            )

        self.L = len(target_aa_seq)
        self.target_aa_seq = target_aa_seq
        self.match_col = match_col  # diagnostic only

        self._aa_to_vocab = aa_to_vocab_map(tokenizer)   # (20,)
        self._aa_argmax = E.argmax(dim=-1)               # (L,) — class indices
        self._vocab_ids = self._aa_to_vocab[self._aa_argmax]  # (L,) — vocab ids
        self._E = E
        self._cursor = 0

    # ----- spec-decoding interface -----

    def remaining(self) -> int:
        return self.L - self._cursor

    def propose(self, K: int) -> list[int]:
        """Return the next K proposed vocab ids. May return fewer than K
        (including zero) if the drafter has been exhausted."""
        K = min(max(K, 0), self.remaining())
        if K == 0:
            return []
        return self._vocab_ids[self._cursor : self._cursor + K].tolist()

    def commit(
        self,
        n_accepted: int,
        verified_token_ids: list[int] | None = None,
    ) -> None:
        if n_accepted < 0:
            raise ValueError("n_accepted must be >= 0")
        _ = verified_token_ids
        self._cursor = min(self._cursor + n_accepted, self.L)

    def reset(self) -> None:
        self._cursor = 0

    # ----- introspection -----

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def emission_argmax_vocab_ids(self) -> torch.LongTensor:
        return self._vocab_ids.clone()


class PrefixAwareHMMDrafter(HMMDrafter):
    """Prefix-aware drafter that re-anchors the HMM at each block boundary.

    The naive drafter aligns the full template once and then proposes by fixed
    position. This variant replaces the already-verified prefix of the template
    with the verifier's accepted residues, realigns that hybrid sequence to the
    HMM, and then reads out the next positions from the refreshed emission
    matrix. That makes proposals depend on the current verified prefix instead
    of only on absolute output position.
    """

    def __init__(self, hmm_path: str | Path, target_aa_seq: str, tokenizer):
        super().__init__(hmm_path, target_aa_seq, tokenizer)
        self._verified_prefix: list[str] = []
        self.reanchor_calls = 0

    def reset(self) -> None:
        super().reset()
        self._verified_prefix = []
        self.reanchor_calls = 0

    def _reanchor_from_prefix(self) -> None:
        prefix = "".join(self._verified_prefix)
        hybrid_seq = prefix + self.target_aa_seq[len(prefix):]
        e_rows, match_col = emission_matrix(
            self.hmm, self._alphabet, hybrid_seq, "prefix-aware"
        )
        E = torch.tensor(e_rows, dtype=torch.float32)
        self.match_col = match_col
        self._E = E
        self._aa_argmax = E.argmax(dim=-1)
        self._vocab_ids = self._aa_to_vocab[self._aa_argmax]
        self.reanchor_calls += 1

    def commit(
        self,
        n_accepted: int,
        verified_token_ids: list[int] | None = None,
    ) -> None:
        if n_accepted < 0:
            raise ValueError("n_accepted must be >= 0")
        if verified_token_ids is None:
            verified_token_ids = []

        chars = [token_id_to_aa(self._tokenizer, token_id) for token_id in verified_token_ids]
        chars = [char for char in chars if char]
        if n_accepted != len(chars):
            raise ValueError(
                f"prefix-aware commit expected {n_accepted} AA tokens, got {len(chars)}"
            )

        if not chars:
            return

        self._verified_prefix.extend(chars)
        self._verified_prefix = self._verified_prefix[:self.L]
        self._cursor = len(self._verified_prefix)
        if self._cursor < self.L:
            self._reanchor_from_prefix()


# ---------------------------------------------------------------------------
# Round-trip check (H2 task 2)
# ---------------------------------------------------------------------------

def round_trip_check(drafter: HMMDrafter, tokenizer) -> str:
    """Scatter E into the full vocab, argmax over the full vocab, decode, and
    compare with the 20-class argmax string. They must agree position-for-
    position — anything else means the 20→V map is not a bijection on the AA
    subset of the vocab and the drafter would propose tokens that the verifier
    can't reconcile.
    """
    expected = "".join(AA_ORDER[i] for i in drafter._aa_argmax.tolist())
    ids = drafter.emission_argmax_vocab_ids.tolist()

    # decode position-by-position so spaces/separators don't confuse us
    decoded_chars: list[str] = []
    for tid in ids:
        s = tokenizer.decode([tid], skip_special_tokens=True)
        s = "".join(s.split())  # drop any spaces the tokenizer inserts
        if len(s) != 1:
            raise AssertionError(
                f"vocab id {tid} decoded to {s!r} (expected exactly one AA letter)"
            )
        decoded_chars.append(s)
    got = "".join(decoded_chars)

    if got != expected:
        for i, (a, b) in enumerate(zip(got, expected)):
            if a != b:
                raise AssertionError(
                    f"round-trip failed at position {i}: full-vocab argmax "
                    f"decoded {a!r}, 20-class argmax expected {b!r}"
                )
        raise AssertionError(
            f"round-trip length mismatch: got {len(got)}, expected {len(expected)}"
        )
    return got


# ---------------------------------------------------------------------------
# KV cache pruning (T5: 4-tuple per layer; only prune self-attn entries)
# ---------------------------------------------------------------------------

def _to_legacy(past):
    if past is None or isinstance(past, tuple):
        return past
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    raise TypeError(f"Unsupported past_key_values type: {type(past)}")


def _prune_self_attn(past, keep_len: int):
    """Prune the self-attn part of a T5 past_key_values tuple to `keep_len`
    positions. Cross-attn (positions 2, 3 in each layer tuple) is encoder-side
    and length-independent of decoder progress — leave it alone."""
    if past is None:
        return None
    out = []
    for layer in past:
        if layer is None:
            out.append(None)
            continue
        if len(layer) == 4:
            k_self, v_self, k_cross, v_cross = layer
            out.append((
                k_self[:, :, :keep_len, :],
                v_self[:, :, :keep_len, :],
                k_cross,
                v_cross,
            ))
        elif len(layer) == 2:
            k_self, v_self = layer
            out.append((
                k_self[:, :, :keep_len, :],
                v_self[:, :, :keep_len, :],
            ))
        else:
            raise ValueError(
                f"Unexpected layer past_key_values tuple length: {len(layer)}"
            )
    return tuple(out)


def _past_self_len(past) -> int:
    if past is None:
        return 0
    return int(past[0][0].shape[2])


# ---------------------------------------------------------------------------
# Leviathan greedy verify loop (H2 task 5)
# ---------------------------------------------------------------------------

def _format_3di(seq: str) -> str:
    return "<fold2AA> " + " ".join(list(seq.lower()))


@torch.inference_mode()
def spec_decode_greedy(
    model,
    tokenizer,
    three_di_seq: str,
    drafter: HMMDrafter,
    K: int,
    device,
    max_new_tokens: int | None = None,
) -> tuple[list[int], dict]:
    """Spec-decode (greedy) the inverse-folding direction with `drafter` as the
    drafter and `model` as the verifier.

    Invariant after each step:
        len(generated) == past_self_len(past_key_values)
    i.e., the self-attn KV cache covers [decoder_start, *generated[:-1]]. The
    last entry of `generated` is `last_token` and will be fed as input[0] of
    the next step (extending the cache by one).

    Returns (generated, stats):
        generated: list of new token ids (EXCLUDES decoder_start; INCLUDES the
                   trailing EOS, matching `model.generate(...)[0][1:]`).
        stats:     {'proposed': total drafter proposals, 'accepted':
                   total accepted proposals, 'steps': spec-decoding steps,
                   'extra_tokens': number of free '+1' verifier tokens emitted}
    """
    if K <= 0:
        raise ValueError(f"K must be >= 1, got {K}")
    drafter.reset()

    # Hard constraint #2: shared vocabulary.
    if drafter.tokenizer_hash != tokenizer_vocab_hash(tokenizer):
        raise AssertionError(
            "drafter and verifier disagree on tokenizer vocab — built against "
            "different tokenizers. (Hard constraint #2 in CLAUDE.md.)"
        )

    L = len(three_di_seq)
    if max_new_tokens is None:
        # Mirror the notebook's enc_dec settings: max_length=L+2, min_length=L+1
        # → output is [decoder_start, *L_AAs, eos], so generated (post-start)
        # has L+1 tokens.
        max_new_tokens = L + 1

    # Encode the 3Di prompt once.
    enc = tokenizer(
        [_format_3di(three_di_seq)], add_special_tokens=True, return_tensors="pt"
    ).to(device)
    encoder_outputs = model.get_encoder()(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        return_dict=True,
    )

    decoder_start = model.config.decoder_start_token_id
    eos_id = model.config.eos_token_id

    generated: list[int] = []
    last_token = decoder_start
    past = None

    n_proposed = 0
    n_accepted = 0
    n_extra = 0
    n_steps = 0

    while len(generated) < max_new_tokens:
        # Cap K by both drafter exhaustion and remaining budget. We can emit at
        # most (max_new_tokens - len(generated)) tokens this step, and one of
        # those is always the verifier's "+1" — so the most proposals we can
        # use is (budget - 1). The verifier path still runs with 0 proposals.
        budget = max_new_tokens - len(generated)
        k_eff = max(0, min(K, drafter.remaining(), budget - 1))
        proposals = drafter.propose(k_eff)
        n_proposed += k_eff
        n_steps += 1

        decoder_input_ids = torch.tensor(
            [[last_token, *proposals]], dtype=torch.long, device=device
        )
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=enc.attention_mask,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        # logits[0, i, :] predicts the token after input position i. With input
        # [last_token, q1, ..., q_k_eff] that's k_eff+1 distributions.
        argmaxes = outputs.logits[0].argmax(dim=-1).tolist()

        # Accept-walk: greedy Leviathan rule = accept iff argmax == proposal.
        committed = 0
        for i in range(k_eff):
            if argmaxes[i] == proposals[i]:
                committed += 1
            else:
                break
        n_accepted += committed

        # Emit: `committed` accepted proposals, then the verifier's argmax at
        # the disagreement (or the free "+1" when committed == k_eff).
        new_tokens = list(proposals[:committed]) + [argmaxes[committed]]
        if committed == k_eff:
            n_extra += 1  # full-accept bonus token

        # Respect EOS and max_new_tokens.
        emit_count = 0
        hit_eos = False
        for t in new_tokens:
            if len(generated) + emit_count >= max_new_tokens:
                break
            emit_count += 1
            if t == eos_id:
                hit_eos = True
                break
        generated.extend(new_tokens[:emit_count])

        # Prune the self-attn KV cache to the verified length. Invariant:
        # keep_len = old_self_len + emit_count == len(generated) (post-extend).
        past_legacy = _to_legacy(outputs.past_key_values)
        keep_len = len(generated)
        past = _prune_self_attn(past_legacy, keep_len)

        # Advance the drafter cursor by the number of new tokens. Capped at
        # k_eff+1 (proposals + the verifier extra), because we never advance
        # past tokens the drafter actually generated.
        verified_aa_ids = aa_token_ids(tokenizer, new_tokens[:emit_count])
        drafter.commit(len(verified_aa_ids), verified_token_ids=verified_aa_ids)

        last_token = generated[-1]
        if hit_eos:
            break

    stats = {
        "proposed": n_proposed,
        "accepted": n_accepted,
        "steps": n_steps,
        "extra_tokens": n_extra,
    }
    return generated, stats


@torch.inference_mode()
def encdec_greedy_reference(model, tokenizer, three_di_seq: str, device) -> list[int]:
    """Reference output: enc-dec greedy, same settings as the notebook's
    `time_encdec`. The return value strips the leading decoder_start token so
    it lines up with `spec_decode_greedy`'s output."""
    L = len(three_di_seq)
    enc = tokenizer(
        [_format_3di(three_di_seq)], add_special_tokens=True, return_tensors="pt"
    ).to(device)
    out = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_length=L + 2,
        min_length=L + 1,
        num_beams=1,
        do_sample=False,
        num_return_sequences=1,
    )
    return out[0, 1:].tolist()


def assert_bit_exact(
    model,
    tokenizer,
    three_di_seq: str,
    aa_seq: str,
    hmm_path: str | Path,
    K: int,
    device,
    label: str = "",
) -> dict:
    """End-to-end bit-exact assertion (hard constraint #1).

    Builds an HMMDrafter for `aa_seq`, runs spec_decode_greedy against `model`
    on `three_di_seq`, computes the enc-dec greedy reference, and asserts the
    two token sequences are identical. Returns spec-decoding stats on success.
    """
    drafter = HMMDrafter(hmm_path, aa_seq, tokenizer)
    spec, stats = spec_decode_greedy(
        model, tokenizer, three_di_seq, drafter, K=K, device=device
    )
    ref = encdec_greedy_reference(model, tokenizer, three_di_seq, device)
    if spec != ref:
        first = next(
            (i for i, (a, b) in enumerate(zip(spec, ref)) if a != b),
            min(len(spec), len(ref)),
        )
        ctx = max(0, first - 3)
        raise AssertionError(
            f"[{label}] spec output diverges from enc-dec greedy at position "
            f"{first}: spec[{ctx}:{first+5}]={spec[ctx:first+5]}  "
            f"ref[{ctx}:{first+5}]={ref[ctx:first+5]}  "
            f"(lengths: spec={len(spec)}, ref={len(ref)})"
        )
    acc_pct = (100.0 * stats["accepted"] / stats["proposed"]) if stats["proposed"] else 0.0
    print(
        f"[{label}] bit-exact OK  L={len(ref)}  K={K}  "
        f"accepted={stats['accepted']}/{stats['proposed']} ({acc_pct:.1f}%)  "
        f"steps={stats['steps']}  free={stats['extra_tokens']}"
    )
    return stats


# ---------------------------------------------------------------------------
# Smoke test (no ProstT5 / no GPU)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal tokenizer stand-in for the smoke test.

    Vocab: AA letters at ids 100..119 (alphabetical), '▁'-prefixed forms at
    120..139 (unused — the bare letter resolves first), '<unk>' at id 0.
    """
    unk_token_id = 0

    def __init__(self):
        self._vocab: dict[str, int] = {"<unk>": 0}
        for i, c in enumerate(AA_ORDER):
            self._vocab[c] = 100 + i
        for i, c in enumerate(AA_ORDER):
            self._vocab[f"▁{c}"] = 120 + i
        self._inv = {v: k for k, v in self._vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, self.unk_token_id)

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for tid in ids:
            tok = self._inv.get(int(tid), "")
            if tok == "<unk>" and skip_special_tokens:
                continue
            out.append(tok.lstrip("▁"))
        return "".join(out)


def _smoke() -> int:
    hmm_path = HERE / "hmm_data" / "PF00535.hmm"
    aa_fasta = HERE / "hmm_data" / "in_family.fasta"
    if not hmm_path.exists() or not aa_fasta.exists():
        sys.exit(
            "Smoke test needs H1 artifacts: run `python build_hmm.py` first to "
            f"produce {hmm_path} and {aa_fasta}."
        )
    aa = "".join(
        line.strip()
        for line in aa_fasta.read_text().splitlines()
        if not line.startswith(">")
    )
    tok = _FakeTokenizer()
    drafter = HMMDrafter(hmm_path, aa, tok)
    print(f"HMMDrafter built: L={drafter.L}  M={drafter.hmm.M}  "
          f"match_col_residues={sum(1 for c in drafter.match_col if c > 0)}")

    # H2 task 1: vocab map shape + dtype
    m = aa_to_vocab_map(tok)
    assert m.shape == (20,) and m.dtype == torch.long, "aa_to_vocab_map shape/dtype"
    print(f"  aa_to_vocab_map: {m.tolist()}")

    # H2 task 2: round-trip 20→V→argmax→decode
    consensus_via_vocab = round_trip_check(drafter, tok)
    print(f"  round-trip OK ({len(consensus_via_vocab)} chars)")

    # propose/commit cursor mechanics
    drafter.reset()
    head = drafter.propose(7)
    assert len(head) == 7, "propose returned wrong count"
    drafter.commit(7)
    assert drafter.cursor == 7
    drafter.commit(drafter.remaining())
    assert drafter.remaining() == 0
    assert drafter.propose(5) == [], "propose past end must return []"
    print("  cursor mechanics OK")

    prefix_aware = PrefixAwareHMMDrafter(hmm_path, aa, tok)
    prefix_head = prefix_aware.propose(4)
    prefix_aware.commit(4, verified_token_ids=prefix_head)
    assert prefix_aware.cursor == 4, "prefix-aware cursor did not advance"
    assert prefix_aware.reanchor_calls == 1, "prefix-aware drafter did not re-anchor"
    assert prefix_aware.propose(4), "prefix-aware proposals unexpectedly empty"
    print("  prefix-aware re-anchoring OK")

    # tokenizer hash stability
    h1 = tokenizer_vocab_hash(tok)
    h2 = tokenizer_vocab_hash(tok)
    assert h1 == h2, "tokenizer hash is not stable"
    different = _FakeTokenizer()
    different._vocab["A"] = 999  # mutate one entry
    h3 = tokenizer_vocab_hash(different)
    assert h1 != h3, "tokenizer hash didn't detect a mutated vocab"
    print(f"  tokenizer hash OK (stable + change-sensitive)  hash[:12]={h1[:12]}")

    print("\nSmoke test passed. Bit-exact end-to-end test still needs ProstT5"
          " + GPU — run `assert_bit_exact(...)` from the notebook.")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="run the no-GPU smoke test (drafter mechanics + round-trip)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.smoke:
        sys.exit(_smoke())
    sys.exit("Nothing to do. Pass --smoke for the no-GPU smoke test, or import "
             "HMMDrafter / PrefixAwareHMMDrafter / spec_decode_greedy / "
             "assert_bit_exact from a notebook.")
