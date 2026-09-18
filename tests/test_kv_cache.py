"""
tests/test_kv_cache.py

Unit tests for KVCache in isolation, independent of the attention module
that consumes it -- covers initialization, incremental update, position
tracking, reset, and per-layer independence.
"""

import torch

from brain.model.transformer.cache import KVCache


def test_cache_starts_empty():
    cache = KVCache(num_layers=3)
    for layer in range(3):
        assert cache.get_seq_length(layer) == 0


def test_prefill_update_shape():
    cache = KVCache(num_layers=1)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    full_k, full_v = cache.update(0, k, v)
    assert full_k.shape == (1, 2, 5, 8)
    assert full_v.shape == (1, 2, 5, 8)
    assert cache.get_seq_length(0) == 5


def test_incremental_update_appends_not_overwrites():
    cache = KVCache(num_layers=1)
    k1 = torch.randn(1, 2, 5, 8)
    v1 = torch.randn(1, 2, 5, 8)
    cache.update(0, k1, v1)

    k2 = torch.randn(1, 2, 1, 8)
    v2 = torch.randn(1, 2, 1, 8)
    full_k, full_v = cache.update(0, k2, v2)

    assert full_k.shape == (1, 2, 6, 8)
    assert torch.equal(full_k, torch.cat([k1, k2], dim=2))
    assert cache.get_seq_length(0) == 6


def test_layers_are_independent():
    cache = KVCache(num_layers=2)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    cache.update(0, k, v)
    assert cache.get_seq_length(0) == 4
    assert cache.get_seq_length(1) == 0  # untouched


def test_reset_clears_all_layers():
    cache = KVCache(num_layers=2)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    cache.update(0, k, v)
    cache.update(1, k, v)
    cache.reset()
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(1) == 0


def test_many_single_token_updates_match_one_batch_update():
    """Ten separate 1-token updates must produce the exact same accumulated
    tensor as a single 10-token update -- this is the property that makes
    step-by-step decoding equivalent to a batch prefill."""
    torch.manual_seed(0)
    full = torch.randn(1, 2, 10, 8)

    batch_cache = KVCache(num_layers=1)
    batch_k, _ = batch_cache.update(0, full, full)

    stepped_cache = KVCache(num_layers=1)
    for i in range(10):
        stepped_k, _ = stepped_cache.update(0, full[:, :, i:i+1, :], full[:, :, i:i+1, :])

    assert torch.equal(batch_k, stepped_k)
