import math
import torch
from brain.data.collator import CausalCollator
from brain.eval.metrics import perplexity_from_loss, preference_accuracy


def test_causal_collator_pads():
    col = CausalCollator(pad_id=0)
    batch = col([[1, 2, 3], [4, 5]])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"][1, 2] == 0
    assert batch["labels"][1, 2] == -100


def test_perplexity():
    assert abs(perplexity_from_loss(0.0) - 1.0) < 1e-6
    assert abs(perplexity_from_loss(math.log(2)) - 2.0) < 1e-6


def test_preference_accuracy():
    assert preference_accuracy([1.0, 0.5], [0.2, 0.9]) == 0.5
