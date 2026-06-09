"""Command-line entry point for ProstT5 assisted generation.

This CLI is intentionally small: it covers the reusable CNN assistant path and
keeps the larger HMM-building workflow in the notebook until that code is ready
to be extracted too.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

from .assistants import CNNAssistantModel
from .cnn import aa_token_ids, load_cnn_head
from .formatting import decode_aa, format_3di


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line arguments for one assisted-generation run."""
    parser = argparse.ArgumentParser(description="Run ProstT5 assisted generation with the CNN drafter.")
    parser.add_argument("--three-di", required=True, help="Raw 3Di sequence, without the <fold2AA> prefix.")
    parser.add_argument("--cnn-checkpoint", required=True, type=Path, help="Path to the AA-CNN model.pt checkpoint.")
    parser.add_argument("--model-name", default="Rostlab/ProstT5_fp16", help="HuggingFace ProstT5 model name.")
    parser.add_argument("--k", default=5, type=float, help="Number of assistant tokens proposed per step.")
    parser.add_argument("--max-extra-tokens", default=2, type=int, help="Extra tokens above input length for EOS handling.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser


def main() -> None:
    """Load ProstT5, wrap the CNN as assistant_model, and print the generated AA."""
    args = build_parser().parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = T5Tokenizer.from_pretrained(args.model_name, do_lower_case=False, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device).eval()
    if dtype == torch.float16:
        model = model.half()

    cnn = load_cnn_head(args.cnn_checkpoint, device)
    aa_ids = aa_token_ids(tokenizer)
    assistant = CNNAssistantModel(
        config=model.config,
        prostt5_encoder=model.get_encoder(),
        cnn_head=cnn,
        aa_token_ids=aa_ids,
        device=device,
    ).to(device).eval()
    assistant.generation_config.num_assistant_tokens = args.k

    encoded = tokenizer(
        [format_3di(args.three_di)],
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_length=len(args.three_di) + args.max_extra_tokens,
            do_sample=False,
            num_beams=1,
            assistant_model=assistant,
        )

    print(decode_aa(tokenizer, output[0])[: len(args.three_di)])


if __name__ == "__main__":
    main()
