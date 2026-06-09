"""CNN drafter components shared by notebooks and command-line runs."""

from pathlib import Path

import torch
import torch.nn as nn


AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


class AACNN(nn.Module):
    """Small ProstT5 encoder-head CNN used for one-shot AA drafting."""

    def __init__(self, num_tokens: int = 20, hidden: int = 32, in_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(hidden, num_tokens, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.classifier(x)
        return x.squeeze(-1).permute(0, 2, 1)


def aa_token_ids(tokenizer, aa_letters: str = AA_LETTERS) -> list[int]:
    """Map the CNN's 20 AA columns to ProstT5 tokenizer IDs."""
    return [tokenizer.encode(f" {aa}", add_special_tokens=False)[0] for aa in aa_letters]


def load_cnn_head(checkpoint_path: str | Path, device, num_tokens: int = 20) -> AACNN:
    """Load the AA-CNN checkpoint with the same architecture as the notebook."""
    cnn = AACNN(num_tokens=num_tokens).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    cnn.load_state_dict(state_dict, strict=True)
    return cnn
