"""Input and output formatting used by ProstT5 inverse folding."""

import torch


def format_3di(seq: str) -> str:
    """Format a raw 3Di string as the ProstT5 inverse-folding prompt."""
    return "<fold2AA> " + " ".join(seq.lower())


def decode_aa(tokenizer, token_ids: torch.Tensor) -> str:
    """Decode generated token IDs into a compact amino-acid sequence."""
    if token_ids.ndim > 1:
        token_ids = token_ids[0]
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text.replace(" ", "").replace("<fold2AA>", "").strip()
