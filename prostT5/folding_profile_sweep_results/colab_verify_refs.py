# Colab (T4) cell — verify that query_3di.fasta IS ProstT5's greedy fold, and
# generate the reference for the one protein missing it (P0DTC9).
#
# If the printed match rate is ~100%, the offline sweep numbers are canonical and
# no further GPU work is needed. Otherwise, this writes refs_3di.json — point
# folding_profile_sweep.py at it (load ref from there instead of query_3di) and
# re-run on CPU.
#
#   !pip -q install transformers sentencepiece torch
#   DATA_DIR = path to the folding_MSA folder (Drive or local)

import json
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration

DATA_DIR = Path("/content/drive/MyDrive/Speculative-Decoding-ProstT5/prostT5/prostT5_benchmarks/folding_MSA")
MODEL = "Rostlab/ProstT5_fp16"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = T5Tokenizer.from_pretrained(MODEL, do_lower_case=False, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL).to(device)
model = model.half() if device.type == "cuda" else model.float()
model.eval()

THREEDI = set("acdefghiklmnpqrstvwy")


def read_fasta1(p):
    seq = []
    for ln in Path(p).read_text().splitlines():
        if not ln.startswith(">"):
            seq.append(ln.strip())
    return "".join(seq)


def fmt_aa(seq):
    return "<AA2fold> " + " ".join(list(seq.upper()))


def decode_3di(ids):
    s = tok.decode(ids, skip_special_tokens=True).replace(" ", "").lower()
    return "".join(c for c in s if c in THREEDI)


@torch.inference_mode()
def greedy_fold(aa):
    L = len(aa)
    enc = tok([fmt_aa(aa)], add_special_tokens=True, return_tensors="pt").to(device)
    out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                         max_length=L + 2, do_sample=False, num_beams=1, use_cache=True)
    return decode_3di(out[0])[:L]


refs, exact, total = {}, 0, 0
for d in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    uid = d.name
    aa = read_fasta1(d / "query_aa.fasta")
    pred = greedy_fold(aa)
    refs[uid] = pred
    q3p = d / "query_3di.fasta"
    if q3p.exists():
        truth = read_fasta1(q3p)
        total += 1
        ident = sum(a == b for a, b in zip(pred, truth)) / max(len(truth), 1)
        exact += (pred == truth)
        print(f"{uid:12} L={len(aa):4} exact={pred == truth!s:5} identity={ident:.3f}")
    else:
        print(f"{uid:12} L={len(aa):4} (no query_3di — generated)")

print(f"\nquery_3di == ProstT5 greedy fold: {exact}/{total} proteins exact")
out = DATA_DIR.parent / "refs_3di.json"      # persists on Drive next to folding_MSA
out.write_text(json.dumps(refs))
print(f"Wrote {out}")
