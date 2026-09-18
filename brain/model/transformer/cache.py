"""
brain/model/transformer/cache.py

A REAL incremental key/value cache.

This directly replaces the Brain Trail prototype's kv_cache_inference.py,
which called an API (`model.backbone(x, use_cache=True, kv_cache=..., 
position_offset=...)`) that TransformerBackbone.forward() never actually
implemented -- confirmed by execution to raise TypeError on the first call.
That file's "O(n) cached generation, proven correct" claim was untested
against real code.

Design: one KVCache instance holds a list of per-layer (key, value) tensor
pairs. Prefill writes the whole prompt's K/V in one shot; each decode step
appends exactly one new token's K/V and returns the full cached tensor
(existing + new) for attention to read. This is the standard pattern used
by every production LLM serving stack.

Kept independent of the model class itself (Engineering Rule: don't create
abstractions the model needs to know about) -- BrainForCausalLM constructs
one and passes it through, but KVCache has zero import of brain.model.
"""

from __future__ import annotations

import torch


class KVCache:
    """Growable per-layer key/value cache for autoregressive decoding.

    Shapes: each layer's key/value tensor is
        (batch, num_kv_heads, seq_len_so_far, head_dim)
    and grows along the seq_len_so_far dimension as tokens are generated.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers

    def update(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's (key, value) for `layer_idx` and return the
        FULL accumulated (key, value) tensors -- i.e. everything attention
        needs to attend over, old and new combined.

        key, value: (batch, num_kv_heads, new_tokens, head_dim)
        """
        if self._keys[layer_idx] is None:
            self._keys[layer_idx] = key
            self._values[layer_idx] = value
        else:
            self._keys[layer_idx] = torch.cat([self._keys[layer_idx], key], dim=2)
            self._values[layer_idx] = torch.cat([self._values[layer_idx], value], dim=2)
        return self._keys[layer_idx], self._values[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Number of positions currently cached (used as the RoPE
        position_offset for the NEXT token to be added)."""
        if self._keys[layer_idx] is None:
            return 0
        return self._keys[layer_idx].shape[2]

    def reset(self) -> None:
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers

    def to(self, device: torch.device) -> "KVCache":
        for i in range(self.num_layers):
            if self._keys[i] is not None:
                self._keys[i] = self._keys[i].to(device)
                self._values[i] = self._values[i].to(device)
        return self


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    NUM_LAYERS, BATCH, KV_HEADS, HEAD_DIM = 2, 1, 2, 8

    cache = KVCache(NUM_LAYERS)
    print(f"Initial seq_length: {cache.get_seq_length()} (expected 0)")

    # Simulate prefill: 5 tokens at once, for layer 0
    prefill_k = torch.randn(BATCH, KV_HEADS, 5, HEAD_DIM)
    prefill_v = torch.randn(BATCH, KV_HEADS, 5, HEAD_DIM)
    full_k, full_v = cache.update(0, prefill_k, prefill_v)
    print(f"After prefill (5 tokens), cached shape: {tuple(full_k.shape)} (expected (1, 2, 5, 8))")
    print(f"seq_length after prefill: {cache.get_seq_length(0)} (expected 5)")

    # Simulate one decode step: 1 new token
    step_k = torch.randn(BATCH, KV_HEADS, 1, HEAD_DIM)
    step_v = torch.randn(BATCH, KV_HEADS, 1, HEAD_DIM)
    full_k, full_v = cache.update(0, step_k, step_v)
    print(f"After 1 decode step, cached shape: {tuple(full_k.shape)} (expected (1, 2, 6, 8))")

    # Correctness: the cached tensor must be the exact concatenation, not a copy/reshape artifact
    reference = torch.cat([prefill_k, step_k], dim=2)
    print(f"Cached K exactly equals concatenation of writes: {torch.equal(full_k, reference)}")

    # Layers are independent
    print(f"Layer 1 still empty while layer 0 has data: {cache.get_seq_length(1) == 0}")

    # Reset works
    cache.reset()
    print(f"After reset, seq_length: {cache.get_seq_length(0)} (expected 0)")
