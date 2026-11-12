from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from brainlab.export.to_safetensors import load_metadata, load_model, save_model, validate_names
from brainlab.model.gpt import TransformerLM
from brainlab.spec.config import load_model_spec

D_MODEL = 512
N_LAYER = 8
N_HEAD = 8
HEAD_DIM = 64
HIDDEN = 1344
VOCAB = 16384
CONTEXT = 1024
PARAM_TOTAL = 33_300_992
METADATA_KEYS = {
    "brainlab.spec_version",
    "brainlab.name",
    "brainlab.config_json",
    "brainlab.param_count",
    "brainlab.source",
    "brainlab.producer_versions",
}


def spec_table() -> dict[str, tuple[int, ...]]:
    table = {
        "token_emb.weight": (VOCAB, D_MODEL),
        "final_norm.weight": (D_MODEL,),
    }
    shapes = {
        "attn_norm.weight": (D_MODEL,),
        "attn_qkv.weight": (3 * D_MODEL, D_MODEL),
        "attn_proj.weight": (D_MODEL, D_MODEL),
        "mlp_norm.weight": (D_MODEL,),
        "mlp_gate.weight": (HIDDEN, D_MODEL),
        "mlp_up.weight": (HIDDEN, D_MODEL),
        "mlp_down.weight": (D_MODEL, HIDDEN),
    }
    for layer in range(N_LAYER):
        for name, shape in shapes.items():
            table[f"layers.{layer}.{name}"] = shape
    return table


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[Path, dict[str, torch.Tensor], dict[str, str]]:
    spec = load_model_spec()
    torch.manual_seed(0)
    model = TransformerLM(spec).eval()
    path = tmp_path_factory.mktemp("weights") / "brainlab-33m.safetensors"
    save_model(model, path, dtype=torch.float32)
    return path, load_file(str(path)), load_metadata(path)


def test_hardcoded_spec_table_matches_the_loaded_checkpoint(exported) -> None:
    _, tensors, _ = exported
    expected = spec_table()
    assert set(tensors) == set(expected), (
        f"only_in_checkpoint={sorted(set(tensors) - set(expected))[:4]} "
        f"only_in_spec={sorted(set(expected) - set(tensors))[:4]}"
    )
    for name, shape in expected.items():
        assert tuple(tensors[name].shape) == shape, f"{name} is {tuple(tensors[name].shape)}, spec says {shape}"


def test_parameter_count_is_exactly_the_declared_total(exported) -> None:
    _, tensors, meta = exported
    total = sum(int(t.numel()) for t in tensors.values())
    assert total == PARAM_TOTAL, f"tensor sum is {total}, spec/weights_format.md says {PARAM_TOTAL}"
    assert meta["brainlab.param_count"] == str(PARAM_TOTAL)
    assert int(json.loads(meta["brainlab.config_json"])["d_model"]) == D_MODEL


def test_no_biases_and_every_two_dimensional_tensor_is_out_by_in(exported) -> None:
    _, tensors, _ = exported
    table = spec_table()
    for name, tensor in tensors.items():
        assert tensor.ndim in (1, 2), f"{name} has {tensor.ndim} dimensions; the format allows only 1-D norms and 2-D matrices"
        if tensor.ndim == 1:
            assert name.endswith("_norm.weight"), f"{name} is a 1-D tensor that is not a norm weight"
        assert tuple(tensor.shape) == table[name], f"{name} is {tuple(tensor.shape)}, spec says {table[name]}"
    assert not [name for name in tensors if name.endswith(".bias")], "biases are not allowed anywhere"
    assert "lm_head.weight" not in tensors, "embeddings are tied, so lm_head.weight must not exist"


def test_metadata_contains_exactly_the_required_keys(exported) -> None:
    _, _, meta = exported
    assert set(meta) == METADATA_KEYS, f"metadata keys drifted: {sorted(set(meta) ^ METADATA_KEYS)}"
    producers = json.loads(meta["brainlab.producer_versions"])
    assert set(producers) == {"torch", "safetensors"}
    assert meta["brainlab.name"] == "brainlab-33m"
    assert meta["brainlab.source"] == "python/brainlab/export/to_safetensors.py"


def test_config_json_round_trips_to_the_same_model(exported) -> None:
    path, _, meta = exported
    from brainlab.spec.config import spec_from_metadata

    rebuilt = spec_from_metadata(meta)
    assert rebuilt.to_dict() == json.loads(meta["brainlab.config_json"])
    assert rebuilt.context_len == CONTEXT and rebuilt.n_heads == N_HEAD and rebuilt.head_dim == HEAD_DIM
    with safe_open(str(path), framework="pt") as handle:
        assert handle.metadata() is not None
    model = load_model(path, device="cpu")
    assert model.spec.to_dict() == rebuilt.to_dict()


def test_loader_rejects_a_param_count_that_disagrees_with_the_shapes(exported) -> None:
    path, tensors, _ = exported
    spec = load_model_spec()
    trimmed = dict(tensors)
    trimmed["token_emb.weight"] = trimmed["token_emb.weight"][: VOCAB - 1]
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_names(trimmed, spec)
    renamed = dict(tensors)
    renamed["lm_head.weight"] = renamed.pop("token_emb.weight")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        validate_names(renamed, spec)
    short = dict(tensors)
    short["final_norm.weight"] = torch.ones(D_MODEL - 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_names(short, spec)


def test_f16_export_keeps_names_shapes_and_stays_representable(tmp_path: Path) -> None:
    spec = load_model_spec()
    torch.manual_seed(0)
    model = TransformerLM(spec).eval()
    path = tmp_path / "brainlab-33m-f16.safetensors"
    save_model(model, path, dtype=torch.float16)
    half = load_file(str(path))
    assert set(half) == set(spec.tensor_shapes())
    for name, tensor in half.items():
        assert tensor.dtype == torch.float16
        assert tuple(tensor.shape) == spec.tensor_shapes()[name]
        assert torch.equal(tensor.float().half(), tensor)


def test_rope_tables_are_not_stored_in_the_checkpoint(exported) -> None:
    _, tensors, _ = exported
    forbidden = [name for name in tensors if "rope" in name or "cos" in name or "sin" in name]
    assert not forbidden, f"derived tensors must not be checkpointed: {forbidden}"
