"""
One DPO training step given policy model (+ optional frozen reference).
"""

from __future__ import annotations

from typing import Optional

import torch

from brain.experimental.dpo.loss import dpo_loss, sequence_logprobs, DPOLossOutput


@torch.no_grad()
def _forward_logps(model, input_ids: torch.Tensor, labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    out = model(input_ids)
    logits = out["logits"] if isinstance(out, dict) else out.logits
    return sequence_logprobs(logits, labels, pad_id=pad_id)


def dpo_train_step(
    policy_model,
    *,
    prompt_ids: torch.Tensor,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    ref_model=None,
    beta: float = 0.1,
    pad_id: int = 0,
) -> DPOLossOutput:
    """
    prompt_ids: (B, P)
    chosen_ids / rejected_ids: (B, R) response tokens only
    Concatenates prompt+response for forward; labels mask prompt with pad_id.
    """
    def pack(resp: torch.Tensor):
        # (B, P+R)
        full = torch.cat([prompt_ids, resp], dim=1)
        labels = full.clone()
        labels[:, : prompt_ids.size(1)] = pad_id
        return full, labels

    pol_c_in, pol_c_lab = pack(chosen_ids)
    pol_r_in, pol_r_lab = pack(rejected_ids)

    out_c = policy_model(pol_c_in)
    out_r = policy_model(pol_r_in)
    logits_c = out_c["logits"] if isinstance(out_c, dict) else out_c.logits
    logits_r = out_r["logits"] if isinstance(out_r, dict) else out_r.logits

    policy_chosen = sequence_logprobs(logits_c, pol_c_lab, pad_id=pad_id)
    policy_rejected = sequence_logprobs(logits_r, pol_r_lab, pad_id=pad_id)

    if ref_model is None:
        # reference-free fallback: treat ref logps as zeros (not ideal but runnable)
        ref_chosen = torch.zeros_like(policy_chosen)
        ref_rejected = torch.zeros_like(policy_rejected)
    else:
        ref_model.eval()
        with torch.no_grad():
            ref_chosen = _forward_logps(ref_model, pol_c_in, pol_c_lab, pad_id)
            ref_rejected = _forward_logps(ref_model, pol_r_in, pol_r_lab, pad_id)

    return dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta
    )
