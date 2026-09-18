"""
DPO loss (Rafailov et al.).

L = -log σ( β * ( log πθ(y_w|x)/πref(y_w|x) - log πθ(y_l|x)/πref(y_l|x) ) )

We compute sequence log-probs under policy and (optional) reference model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DPOLossOutput:
    loss: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    accuracy: torch.Tensor  # fraction where chosen > rejected under implicit reward


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    logits: (B, T, V), labels: (B, T)
    Returns per-sequence sum logprob of labels[1:] given logits[:-1] style
    if labels are the full sequence including prompt — caller should pass
    response-only labels with prompt positions set to pad_id for ignore.
    """
    # shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != pad_id).float()
    # sum over tokens
    return (gathered * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> DPOLossOutput:
    """
    All logps: (B,) sequence sums.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    acc = (chosen_reward > rejected_reward).float().mean()
    return DPOLossOutput(
        loss=loss,
        chosen_reward=chosen_reward.mean(),
        rejected_reward=rejected_reward.mean(),
        accuracy=acc,
    )
