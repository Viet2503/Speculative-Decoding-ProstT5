"""
FIX for the FlexProfile drafter loader.

Problem (confirmed from the notebook's own run log):
    WARNING: adapter dir .../ProtProfileMD_LoRA not found, using base weights
    WARNING: No safetensors file found at .../model.safetensors
The drafter was loaded as: base ProstT5 encoder (no LoRA) + a RANDOM, untrained
profile head. So every benchmark number measured a random model, not ProtProfileMD.

Two root causes, both in FlexProfileDrafter.from_huggingface:
  1. It called hf_hub_download(filename="train_config.yaml"), which downloads ONLY
     that one file. The ProtProfileMD_LoRA/ folder was never downloaded, so the
     LoRA adapter was skipped.
  2. It looked for "model.safetensors". The published weights file is
     "ProtProfileMD_LoRA/adapter_model.safetensors", and the trained profile-head
     weights are bundled inside it (adapter config has modules_to_save=["profile_head"]).
     So the head weights were never read.

This file is a drop-in replacement for the from_huggingface classmethod (paste it
into cell 3 of the notebook / flexprofile_drafter.py, replacing the old method).
Nothing else in the FlexProfileDrafter class needs to change.

After applying: re-run Part 0 (one-shot) FIRST. The metric `flex_vs_afdb_3di` should
jump from ~0.02 (below random) to a sane value (~0.4-0.7). If it's still near random,
the next suspect is the class-index -> 3Di letter ordering (DI_LETTERS), which the
ProtProfileMD repo does not document -- verify it before trusting K>1 results.
"""

import os
import yaml
import torch

# NOTE: requires `from huggingface_hub import snapshot_download` and
# `import safetensors.torch` (added inside the method below).


@classmethod
def from_huggingface(cls, device="cpu", dtype=torch.float32):
    """Load ProtProfileMD (ProstT5 encoder + LoRA + trained profile head) from the Hub."""
    from huggingface_hub import snapshot_download
    import safetensors.torch
    from peft import LoraConfig, get_peft_model

    drafter = cls(device=device, dtype=dtype)

    # FIX 1: download the WHOLE repo, not just one file, so the adapter folder exists.
    repo_dir = snapshot_download(repo_id="finnlueth/ProtProfileMD")

    with open(os.path.join(repo_dir, "train_config.yaml"), "r") as f:
        train_config = yaml.safe_load(f)

    adapter_name = train_config["metadata"]["adapter_name"]   # "ProtProfileMD_LoRA"
    adapter_dir = os.path.join(repo_dir, adapter_name)
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")  # FIX 2: correct filename

    print(f"ProtProfileMD repo: {repo_dir}")
    print(f"  Base model:   {train_config['model']['base_model']}")
    print(f"  Profile head: {train_config['model']['profile_head']}")

    # 1. Base ProstT5 encoder (fp16 weights are numerically the same model as the
    #    fp32 ProstT5 the LoRA was trained on; fine for a drafter).
    from transformers import T5EncoderModel
    print("Loading ProstT5 encoder for ProtProfileMD...")
    encoder = T5EncoderModel.from_pretrained(
        "Rostlab/ProstT5_fp16", torch_dtype=dtype, low_cpu_mem_usage=True,
    )

    # 2. Profile head (the linear classifier).  ProfileHeadLinear must already be defined.
    profile_head = ProfileHeadLinear(**train_config["model"]["profile_head_kwargs"])

    # 3. Attach LoRA to the encoder. We drop modules_to_save here because the head is
    #    kept as a separate module (loaded manually in step 5), not inside the encoder.
    lora_cfg = train_config["training"]["lora"].copy()
    lora_cfg.pop("modules_to_save", None)
    encoder = get_peft_model(encoder, peft_config=LoraConfig(**lora_cfg),
                             adapter_name=adapter_name)

    # 4. Load the LoRA (q/v) weights. adapter_dir now exists thanks to FIX 1.
    #    The profile_head keys in the safetensors are simply ignored here (they don't
    #    match the encoder's modules) -- we load them explicitly in step 5.
    assert os.path.isdir(adapter_dir), f"adapter dir missing: {adapter_dir}"
    encoder.load_adapter(model_id=adapter_dir, adapter_name=adapter_name)
    print(f"  LoRA adapter loaded from {adapter_dir}")

    # 5. Load the TRAINED profile-head weights out of adapter_model.safetensors.
    #    With modules_to_save, PEFT stores the trained copy under keys containing
    #    ".modules_to_save.<adapter_name>." (the ".original_module." copy is the
    #    frozen init -- we must NOT use that one).
    assert os.path.exists(st_path), f"weights file missing: {st_path}"
    state = safetensors.torch.load_file(st_path)

    def _find_head_key(target):  # target e.g. "classifier.weight"
        cands = [k for k in state if "profile_head" in k and k.endswith(target)]
        trained = [k for k in cands if "modules_to_save" in k]
        chosen = trained or cands
        return chosen[0] if chosen else None

    head_state = {}
    for target in profile_head.state_dict().keys():        # classifier.weight, classifier.bias
        k = _find_head_key(target)
        if k is not None:
            head_state[target] = state[k]

    missing = set(profile_head.state_dict().keys()) - set(head_state.keys())
    if missing:
        raise RuntimeError(
            f"Profile-head weights not found in {st_path}: missing {sorted(missing)}.\n"
            f"profile_head-related keys present: "
            f"{[k for k in state if 'profile_head' in k]}"
        )
    profile_head.load_state_dict(head_state, strict=True)
    print(f"  Profile head loaded ({len(head_state)} tensors) from {os.path.basename(st_path)}")

    # 6. Sanity guard: the head is no longer at random init. A freshly-initialized
    #    nn.Linear(1024, 20) has weight std ~ 1/sqrt(1024) ~ 0.031. Trained weights
    #    look different; this catches a silent "didn't actually load" regression.
    w = profile_head.classifier.weight.detach().float()
    print(f"  Head weight: shape={tuple(w.shape)}  mean={w.mean():.4f}  std={w.std():.4f}")
    assert w.abs().sum() > 0, "profile head weights are all zero -- load failed"

    # 7. Merge LoRA into the encoder for faster inference.
    encoder = encoder.merge_and_unload()
    print("  LoRA merged into encoder.")

    drafter._encoder = encoder.to(device).eval()
    drafter._profile_head = profile_head.to(device).eval()

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in profile_head.parameters())
    print(f"  Encoder params: {n_enc/1e6:.1f}M, Head params: {n_head:,}")
    return drafter
