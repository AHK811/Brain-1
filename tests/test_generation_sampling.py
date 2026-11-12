from __future__ import annotations

import pytest
import torch

from brainlab.eval.generate import keep_top_k, keep_top_p

MIXED = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]])


def kept_indices(logits: torch.Tensor, p: float) -> list[int]:
    masked = keep_top_p(logits, p)[0]
    return [int(index) for index in torch.nonzero(masked > float("-inf")).flatten()]


def nucleus(logits: torch.Tensor, p: float) -> list[int]:
    order = logits[0].argsort(descending=True).tolist()
    probabilities = logits[0].softmax(-1)
    total = 0.0
    chosen: list[int] = []
    for index in order:
        if total >= p:
            break
        chosen.append(int(index))
        total += float(probabilities[index])
    return sorted(chosen)


def test_top_p_keeps_the_smallest_prefix_reaching_p() -> None:
    for p in (0.1, 0.5, 0.665, 0.9, 0.99):
        assert kept_indices(MIXED, p) == nucleus(MIXED, p), f"p={p}"


def test_top_p_of_a_sharp_distribution_keeps_one_token() -> None:
    assert kept_indices(MIXED, 0.5) == [0], "the top token alone already exceeds p"


def test_negative_logits_are_not_masked_away_entirely() -> None:
    all_negative = -MIXED
    assert kept_indices(all_negative, 0.5) == nucleus(all_negative, 0.5)
    assert kept_indices(all_negative, 0.5) == [7], "sorting is by logit, so the max sits at index 7"


def test_kept_tokens_stay_in_their_original_positions() -> None:
    offcentre = torch.tensor([[0.0, 5.0, 1.0]])
    assert kept_indices(offcentre, 0.5) == [1]
    masked = keep_top_p(offcentre, 0.5)
    assert float(masked[0][1]) == 5.0, "scatter must restore the original layout, not permute values"


def test_top_p_boundaries() -> None:
    assert torch.equal(keep_top_p(MIXED, 1.0), MIXED)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="top_p must be"):
            keep_top_p(MIXED, bad)


def test_kept_mass_is_at_least_p() -> None:
    probabilities = MIXED[0].softmax(-1)
    for p in (0.3, 0.7, 0.95):
        total = float(sum(probabilities[i] for i in kept_indices(MIXED, p)))
        assert total >= p - 1e-6, f"p={p} kept only {total}"


def test_top_k_keeps_the_requested_count() -> None:
    for k in (1, 3, 8):
        assert len(kept_indices(keep_top_k(MIXED, k), 1.0)) == k


@pytest.mark.skipif(torch.cuda.is_available(), reason="this host has a GPU, so the guard cannot fire")
def test_explicit_cuda_request_is_not_silently_run_on_cpu(tmp_path) -> None:
    from brainlab.eval.generate import main

    with pytest.raises(SystemExit, match="CUDA is unavailable"):
        main(["--model", str(tmp_path / "unused.safetensors"), "--tokenizer-dir", str(tmp_path), "--device", "cuda"])
