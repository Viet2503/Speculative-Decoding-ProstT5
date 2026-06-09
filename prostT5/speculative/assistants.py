"""HuggingFace assistant_model wrappers for ProstT5 speculative decoding."""

import torch
from transformers import GenerationConfig, PreTrainedModel, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput


class CNNAssistantModel(PreTrainedModel):
    """Wrap the enc-CNN path as a HuggingFace-compatible assistant model.

    The CNN drafter is prefix-independent: for a fixed encoder input it produces
    one full-length matrix of AA logits. The wrapper caches those logits and
    serves the positions requested by HuggingFace's assisted-generation loop.
    """

    config_class = T5Config

    def __init__(self, config, prostt5_encoder, cnn_head, aa_token_ids, device):
        super().__init__(config)
        self._encoder = prostt5_encoder
        self._cnn = cnn_head
        self._aa_token_ids = aa_token_ids
        self._device = device
        self.config.is_encoder_decoder = True
        self.config.decoder_start_token_id = config.decoder_start_token_id
        self.generation_config = GenerationConfig(
            num_assistant_tokens=5,
            num_assistant_tokens_schedule="constant",
            do_sample=False,
            max_length=3000,
            return_dict_in_generate=True,
            output_scores=True,
        )
        self._cached_logits = None
        self._cached_input_hash = None

    def _compute_cnn_logits(self, encoder_outputs):
        """Run the CNN once and expand its 20 AA logits to full vocabulary size."""
        hidden = encoder_outputs[0]
        h = hidden[:, 1:-1, :]
        logits_20 = self._cnn(h.float())[0]

        vocab_size = self.config.vocab_size
        full_logits = torch.full(
            (logits_20.shape[0], vocab_size),
            -100.0,
            device=logits_20.device,
        )
        for i, token_id in enumerate(self._aa_token_ids):
            full_logits[:, token_id] = logits_20[:, i]
        return full_logits

    def get_encoder(self):
        return self._encoder

    def _validate_model_kwargs(self, model_kwargs):
        return

    def prepare_inputs_for_generation(self, decoder_input_ids, encoder_outputs=None, **kwargs):
        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
        }

    def forward(self, decoder_input_ids=None, encoder_outputs=None, **kwargs):
        """Return cached CNN logits for the decoder positions being queried."""
        if encoder_outputs is not None:
            hidden_id = id(encoder_outputs[0])
            if self._cached_input_hash != hidden_id:
                self._cached_logits = self._compute_cnn_logits(encoder_outputs)
                self._cached_input_hash = hidden_id

        seq_len = decoder_input_ids.shape[1]
        n_positions = seq_len
        if self._cached_logits is not None:
            cached_len = self._cached_logits.shape[0]
            n_to_return = min(n_positions, cached_len)
            logits = self._cached_logits[:n_to_return].unsqueeze(0)
            if n_to_return < n_positions:
                pad = torch.full(
                    (1, n_positions - n_to_return, logits.shape[-1]),
                    -100.0,
                    device=logits.device,
                )
                logits = torch.cat([logits, pad], dim=1)
        else:
            logits = torch.zeros(
                (1, n_positions, self.config.vocab_size),
                device=self._device,
            )

        return Seq2SeqLMOutput(logits=logits)


class ProfileHMMAssistantModel(PreTrainedModel):
    """Wrap static or prefix-aware Profile HMM drafters for assisted generation.

    Bind a per-protein drafter with set_drafter() before generation. Static HMM
    drafters are expected to expose get_draft_logits(); prefix-aware drafters are
    expected to expose emission_for_prefix().
    """

    config_class = T5Config

    def __init__(self, config, prostt5_encoder, aa_token_ids, device):
        super().__init__(config)
        self._encoder = prostt5_encoder
        self._aa_ids = torch.tensor(aa_token_ids, device=device, dtype=torch.long)
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

    def set_drafter(self, drafter):
        """Bind the per-protein drafter used for the next generation."""
        self._active = drafter

    def get_encoder(self):
        return self._encoder

    def _validate_model_kwargs(self, model_kwargs):
        return

    def prepare_inputs_for_generation(self, decoder_input_ids, encoder_outputs=None, **kwargs):
        return {"decoder_input_ids": decoder_input_ids, "encoder_outputs": encoder_outputs}

    def forward(self, decoder_input_ids=None, encoder_outputs=None, **kwargs):
        seq_len = decoder_input_ids.shape[1]
        pos = seq_len - 1
        vocab = self.config.vocab_size
        full = torch.full((1, seq_len, vocab), -100.0, device=decoder_input_ids.device)

        row20 = None
        active = self._active
        if active is not None:
            if hasattr(active, "emission_for_prefix"):
                prefix = decoder_input_ids[0, 1:].tolist()
                row20 = torch.from_numpy(active.emission_for_prefix(prefix)).to(self._device)
            else:
                draft_logits = active.get_draft_logits(pos, None, None)
                row20 = draft_logits[0] if draft_logits.shape[0] > 0 else None

        if row20 is not None:
            full[0, -1, self._aa_ids] = row20.to(full.dtype)
        return Seq2SeqLMOutput(logits=full)
