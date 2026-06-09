"""Reusable speculative-decoding helpers for ProstT5.

The original notebooks remain unchanged. This package mirrors the notebook
workflow in importable Python modules so experiments can be run without copying
large notebook cells around.
"""

from .assistants import CNNAssistantModel, ProfileHMMAssistantModel
from .cnn import AACNN, aa_token_ids, load_cnn_head
from .formatting import decode_aa, format_3di

__all__ = [
    "AACNN",
    "CNNAssistantModel",
    "ProfileHMMAssistantModel",
    "aa_token_ids",
    "decode_aa",
    "format_3di",
    "load_cnn_head",
]
