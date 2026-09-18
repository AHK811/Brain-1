"""
Phase E — Multimodal SFT training steps.

Strategies:
  1. Text-only SFT on build_prompt() strings (works immediately with ingest/OCR context)
  2. Vision projector + LM: feed images through VisionTower while training
  3. DPO on MMPreferenceExample (text side; images optional)

Full trainers belong in notebooks; these are correct building blocks.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from brain.experimental.mm_sft.data import MMSFTExample, MMPreferenceExample
from brain.experimental.dpo.train_step import dpo_train_step


def encode_batch(
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int = 1024,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return input_ids, labels (pad → -100 for labels)."""
    ids_list = []
    for t in texts:
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(t, add_special_tokens=True)
            if isinstance(ids, dict):
                ids = ids.get("input_ids", ids)
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
        else:
            ids = list(t)
        ids = list(ids)[:max_length]
        ids_list.append(ids)
    max_len = max(len(x) for x in ids_list) if ids_list else 0
    b = len(ids_list)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return input_ids, labels


def mm_sft_step(
    model,
    tokenizer,
    batch: Sequence[MMSFTExample],
    optimizer: torch.optim.Optimizer,
    *,
    mm_model=None,
    max_length: int = 1024,
    device: str = "cpu",
) -> float:
    """
    One SFT step. If mm_model + examples have images, loads tensors when possible.
    """
    prompts = [ex.build_prompt() for ex in batch]
    input_ids, labels = encode_batch(tokenizer, prompts, max_length=max_length)
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    # replace -100 with pad for models that don't support ignore_index
    labels_for_model = labels.clone()
    labels_for_model[labels_for_model < 0] = 0

    optimizer.zero_grad(set_to_none=True)
    images = None
    if mm_model is not None and any(ex.images for ex in batch):
        try:
            from brain.experimental.multimodal.vision.preprocess import load_image_batch
            paths = []
            for ex in batch:
                paths.append(ex.images[0] if ex.images else None)
            # only if all exist
            if all(p and __import__("pathlib").Path(p).exists() for p in paths):
                images = load_image_batch(paths).to(device)
                out = mm_model(input_ids, images=images, labels=labels_for_model)
            else:
                out = model(input_ids, labels=labels_for_model)
        except Exception:
            out = model(input_ids, labels=labels_for_model)
    else:
        out = model(input_ids, labels=labels_for_model)

    loss = out["loss"] if isinstance(out, dict) else out.loss
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def mm_dpo_step(
    policy_model,
    tokenizer,
    batch: Sequence[MMPreferenceExample],
    optimizer: torch.optim.Optimizer,
    *,
    ref_model=None,
    beta: float = 0.1,
    max_length: int = 512,
    device: str = "cpu",
    pad_id: int = 0,
) -> float:
    """DPO step on multimodal preference pairs (text primary)."""
    prompts, chosens, rejecteds = [], [], []
    for ex in batch:
        prompts.append(ex.prompt)
        chosens.append(ex.chosen)
        rejecteds.append(ex.rejected)

    def tok_texts(texts, add_special=True):
        ids = []
        for t in texts:
            x = tokenizer.encode(t, add_special_tokens=add_special)
            if isinstance(x, dict):
                x = x["input_ids"]
            ids.append(list(x)[:max_length])
        m = max(len(i) for i in ids)
        out = torch.full((len(ids), m), pad_id, dtype=torch.long)
        for i, row in enumerate(ids):
            out[i, : len(row)] = torch.tensor(row)
        return out.to(device)

    prompt_ids = tok_texts(prompts)
    # responses only — approximate: encode answer alone
    chosen_ids = tok_texts(chosens, add_special=False)
    rejected_ids = tok_texts(rejecteds, add_special=False)

    optimizer.zero_grad(set_to_none=True)
    out = dpo_train_step(
        policy_model,
        prompt_ids=prompt_ids,
        chosen_ids=chosen_ids,
        rejected_ids=rejected_ids,
        ref_model=ref_model,
        beta=beta,
        pad_id=pad_id,
    )
    out.loss.backward()
    optimizer.step()
    return float(out.loss.detach())
