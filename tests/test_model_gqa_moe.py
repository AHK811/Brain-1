from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from brainlab.model.gpt import TransformerLM
from brainlab.model.layers import SparseMoE, apply_rope


def manual_grouped_attention(block, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    attn = block.attn
    b, t, d = x.shape
    q, k, v = attn.qkv(x).split((d, attn.kv_dim, attn.kv_dim), dim=-1)
    q = apply_rope(q.view(b, t, attn.n_heads, attn.head_dim).transpose(1, 2), cos, sin)
    k = apply_rope(k.view(b, t, attn.n_kv_heads, attn.head_dim).transpose(1, 2), cos, sin)
    v = v.view(b, t, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    group = attn.n_heads // attn.n_kv_heads
    k = k.repeat_interleave(group, dim=1).float()
    v = v.repeat_interleave(group, dim=1).float()
    scores = q.float() @ k.transpose(-2, -1) * attn.scale
    scores = scores.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    out = (probabilities @ v).to(q.dtype)
    return attn.proj(out.transpose(1, 2).reshape(b, t, d))


def test_grouped_query_attention_matches_an_independent_repeat_impl(smoke_spec) -> None:
    torch.manual_seed(0)
    model = TransformerLM(smoke_spec).eval()
    ids = torch.randint(0, smoke_spec.vocab_size, (2, 9))
    with torch.no_grad():
        x = model.drop(model.token_emb(ids))
        cos, sin = model.rope(9, x.device, x.dtype)
        for block in model.blocks:
            normalized = block.attn_norm(x)
            produced, _ = block.attn(normalized, cos, sin, None)
            expected = manual_grouped_attention(block, normalized, cos, sin)
            worst = (produced - expected).abs().max().item()
            reference = expected.abs().max().item()
            assert worst < 1e-5 * max(reference, 1.0), f"gqa head mapping disagrees, worst {worst}"
            x = block(x, cos, sin)[0]


def test_kv_projection_is_narrower_than_q(smoke_spec) -> None:
    model = TransformerLM(smoke_spec)
    attn = model.blocks[0].attn
    assert attn.kv_dim == smoke_spec.n_kv_heads * smoke_spec.head_dim
    assert attn.qkv.out_features == smoke_spec.d_model + 2 * attn.kv_dim
    assert attn.n_heads != attn.n_kv_heads, "the smoke spec must exercise the grouped case"
    k_rows = attn.qkv.weight[smoke_spec.d_model : smoke_spec.d_model + attn.kv_dim]
    v_rows = attn.qkv.weight[smoke_spec.d_model + attn.kv_dim :]
    assert k_rows.shape == v_rows.shape == (attn.kv_dim, smoke_spec.d_model)


def test_moe_routes_every_token_to_exactly_two_experts_without_dropping(smoke_spec) -> None:
    torch.manual_seed(0)
    model = TransformerLM(smoke_spec).eval()
    moe = model.blocks[0].mlp
    assert isinstance(moe, SparseMoE)
    x = torch.randn(3, 7, smoke_spec.d_model)
    flat = x.reshape(-1, smoke_spec.d_model)
    with torch.no_grad():
        out, balance = moe(x)
        scores = flat @ moe.router.weight.t()
        selected = scores.argsort(dim=-1, descending=True, stable=True)[:, : moe.experts_per_token]
        weights = scores.gather(1, selected).softmax(dim=-1)
        rebuilt = torch.zeros_like(flat)
        for slot in range(moe.experts_per_token):
            for row in range(flat.shape[0]):
                rebuilt[row] += weights[row, slot] * moe.experts[selected[row, slot]](flat[row : row + 1])[0]
        assert torch.allclose(out.reshape(-1, smoke_spec.d_model), rebuilt, atol=1e-5)
    assert selected.shape == (flat.shape[0], 2)
    assert bool(balance == 0), "the load-balancing loss is a training quantity"


def test_moe_aux_loss_is_layer_averaged_and_only_added_while_training(smoke_spec) -> None:
    torch.manual_seed(0)
    model = TransformerLM(smoke_spec)
    ids = torch.randint(0, smoke_spec.vocab_size, (2, 11))
    targets = torch.randint(0, smoke_spec.vocab_size, (2, 11))
    model.train()
    _, balance = model._hidden(ids)
    balance = balance.detach()
    model.eval()
    with torch.no_grad():
        _, eval_loss = model(ids, targets)
    model.train()
    _, train_loss = model(ids, targets)
    assert 0.0 <= float(balance) <= float(smoke_spec.n_experts), f"layer-mean balance out of range: {balance}"
    plain = float(train_loss.detach()) - smoke_spec.router_aux_loss_coef * float(balance)
    assert math.isclose(plain, float(eval_loss), rel_tol=1e-4, abs_tol=1e-6), (
        "train loss must be the eval loss plus coef times the layer-mean balance"
    )


def test_router_breaks_ties_by_expert_index(smoke_spec) -> None:
    torch.manual_seed(1)
    model = TransformerLM(smoke_spec).eval()
    moe: SparseMoE = model.blocks[0].mlp
    moe.router.weight.data.zero_()
    x = torch.randn(1, 2, smoke_spec.d_model)
    with torch.no_grad():
        out, _ = moe(x)
        expected = (moe.experts[0](x) + moe.experts[1](x)) / 2
        unselected = moe.experts[2](x)
    assert torch.allclose(out, expected, atol=1e-6), "equal router scores must select the lowest indices"
    assert unselected.abs().max().item() > 0, "the tie test needs experts that differ"


def test_cached_decoding_reproduces_the_full_forward(smoke_spec) -> None:
    torch.manual_seed(0)
    model = TransformerLM(smoke_spec).eval()
    ids = torch.randint(0, smoke_spec.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(ids)
        chunked, cache = model.forward_cached(ids[:, :5])
        rest, cache = model.forward_cached(ids[:, 5:], cache=cache)
        logits = torch.cat((chunked, rest), dim=1)
        assert logits.shape == full.shape
        worst = (logits - full).abs().max().item()
        assert worst < 2e-4, f"cached attention drifts from the full pass by {worst}"
        step, single = model.forward_cached(ids[:, :11])
        last, single = model.forward_cached(ids[:, 11:], cache=single)
        assert (step - full[:, :11]).abs().max().item() < 2e-4, "the cached prefix must match the full pass"
        assert (last - full[:, 11:12]).abs().max().item() < 2e-4, "token-at-a-time decode must match the full pass"
        assert all(kv[0].shape == (1, model.spec.n_kv_heads, 12, model.spec.head_dim) for kv in single)


def test_cached_forward_refuses_a_train_mode_and_an_overlong_context(smoke_spec) -> None:
    model = TransformerLM(smoke_spec)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model.forward_cached(torch.ones(1, 3, dtype=torch.long))
    model.eval()
    with pytest.raises(ValueError, match="context_len"):
        model.forward_cached(torch.ones(1, smoke_spec.context_len + 1, dtype=torch.long))
    short = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)) for _ in model.blocks]
    with pytest.raises(ValueError, match="cache"):
        model.forward_cached(torch.ones(1, 2, dtype=torch.long), cache=short[:1])
    with pytest.raises(ValueError, match="cache"):
        model.forward_cached(torch.ones(1, 2, dtype=torch.long), cache=short)


def test_parameter_count_and_shapes_come_from_the_spec(smoke_spec) -> None:
    model = TransformerLM(smoke_spec)
    assert model.num_params() == smoke_spec.param_count
    stored = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    assert stored["lm_head.weight"] == (smoke_spec.vocab_size, smoke_spec.d_model)
    assert stored["blocks.0.attn.qkv.weight"] == (smoke_spec.qkv_out, smoke_spec.d_model)
    assert stored["blocks.0.mlp.experts.3.down.weight"] == (smoke_spec.d_model, smoke_spec.mlp_hidden)
