from __future__ import annotations

import pytest
import torch

from brainlab.model.gpt import TransformerLM
from brainlab.train.optimizer import Muon, build_optimizers, split_params


def test_groups_partition_the_model_exactly(spec) -> None:
    model = TransformerLM(spec)
    layers = range(spec.n_layers)
    groups = split_params(model)
    expected_matrix = {
        f"blocks.{index}.attn.{name}.weight" for index in layers for name in ("qkv", "proj")
    } | {
        f"blocks.{index}.mlp.experts.{slot}.{name}.weight"
        for index in layers
        for slot in range(spec.n_experts)
        for name in ("gate", "up", "down")
    }
    assert {name for name, _ in groups["matrix"]} == expected_matrix
    assert {name for name, _ in groups["embedding"]} == {"token_emb.weight", "lm_head.weight"}
    assert {name for name, _ in groups["router"]} == {
        f"blocks.{index}.mlp.router.weight" for index in layers
    }
    assert {name for name, _ in groups["norm"]} == {"final_norm.weight"} | {
        f"blocks.{index}.{name}.weight" for index in layers for name in ("attn_norm", "mlp_norm")
    }
    handles = [id(param) for bucket in groups.values() for _, param in bucket]
    total = len(list(model.parameters()))
    assert len(handles) == len(set(handles)) == total, "a parameter is in two groups or in none"


def test_block_matrices_are_optimised_by_muon(spec) -> None:
    model = TransformerLM(spec)
    names = {id(param): name for name, param in model.named_parameters()}
    muon, adamw = build_optimizers(model, mode="muon")
    muon_names = {names[id(p)] for group in muon.param_groups for p in group["params"]}
    adam_decay = {names[id(p)]: group["weight_decay"] for group in adamw.param_groups for p in group["params"]}
    adam_names = set(adam_decay)
    expected = {
        "qkv": spec.n_layers,
        "proj": spec.n_layers,
        "gate": spec.n_layers * spec.n_experts,
        "up": spec.n_layers * spec.n_experts,
        "down": spec.n_layers * spec.n_experts,
    }
    for suffix, count in expected.items():
        hit = [name for name in muon_names if name.endswith("." + suffix + ".weight")]
        assert len(hit) == count, f"{suffix} must be in Muon on every layer, D017"
    assert not (muon_names & adam_names), "the two optimizers must not share a parameter"
    assert {"token_emb.weight", "lm_head.weight"} <= adam_names
    assert len(adam_names) == 3 + 2 * spec.n_layers + spec.n_layers, "AdamW owns the head, embedding, routers and norms"
    norms = [name for name in adam_names if name.endswith("_norm.weight")]
    assert len(norms) == 2 * spec.n_layers + 1, "every norm gain reaches AdamW"
    assert all(adam_decay[name] == 0.0 for name in norms), "norms must not decay"
    assert all(adam_decay[name] > 0.0 for name in ("token_emb.weight", "lm_head.weight")), "the head must decay"


def test_muon_counts_a_non_finite_update_instead_of_hiding_it() -> None:
    param = torch.zeros(4, 4, requires_grad=True)
    optim = Muon([param], lr=0.1, weight_decay=0.0)
    param.grad = torch.full((4, 4), float("inf"))
    optim.step()
    assert optim.skipped_updates() == 1
    assert torch.equal(param, torch.zeros(4, 4)), "a poisoned update must not move the parameter"
    param.grad = torch.ones(4, 4)
    optim.step()
    assert optim.skipped_updates() == 1, "a healthy step must not raise the counter"
    assert not torch.equal(param, torch.zeros(4, 4))


def test_muon_does_not_decay_a_parameter_it_refused_to_update() -> None:
    param = torch.ones(4, 4, requires_grad=True)
    optim = Muon([param], lr=0.5, weight_decay=0.2, momentum=0.0)
    param.grad = torch.full((4, 4), float("nan"))
    optim.step()
    assert optim.skipped_updates() == 1
    assert torch.equal(param, torch.ones(4, 4)), "a discarded update must not shrink the parameter as well"
    param.grad = torch.ones(4, 4)
    optim.step()
    assert optim.skipped_updates() == 1
    assert torch.all(param < 1.0), "the poisoned momentum buffer must not freeze the parameter forever"


@pytest.mark.parametrize("mode", ["muon", "adamw"])
def test_both_modes_cover_the_model(spec, mode: str) -> None:
    model = TransformerLM(spec)
    optims = build_optimizers(model, mode=mode)
    covered = {id(p) for optim in optims for group in optim.param_groups for p in group["params"]}
    assert covered == {id(p) for p in model.parameters()}
