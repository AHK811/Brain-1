"""
brain/model/brain_model.py

BrainForCausalLM assembles the full stack:

    Input IDs -> TokenEmbedding -> TransformerBlock x N -> Final RMSNorm
              -> LM Head (tied to TokenEmbedding's weight) -> Logits

This class knows about torch and brain.core.config, and NOTHING else --
no FastAPI, no database, no tokenizer library types (it takes/returns
plain integer tensors). Any code that imports fastapi, sqlalchemy, or
similar inside this file is a bug, not a feature; the audit's Step 5
boundary is enforced by discipline, not by a runtime check, so keep it
that way when this file gets touched later.

Also home to:
  - parameter counting (Step 13 of the implementation prompt: don't just
    CLAIM "27M parameters", calculate and report it)
  - checkpoint save/load that bundles config + tokenizer info + training
    metadata, fixing audit Problem #3 (Brain Trail saved bare state_dict()
    with no way to know what architecture it belonged to)
  - a generate() that supports both the plain full-recompute path (for
    correctness comparison / no-cache use) and the real KV-cache path
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.core.config import BrainConfig
from brain.model.embeddings import TokenEmbedding, LearnedPositionalEmbedding
from brain.model.initialization import init_weights
from brain.model.transformer.block import TransformerBlock
from brain.model.transformer.cache import KVCache
from brain.model.transformer.normalization import RMSNorm
from brain.model.transformer.rope import RotaryEmbedding


class BrainForCausalLM(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            # Real parameter sharing -- the SAME tensor object is used for
            # both, not two separately-initialized tensors that happen to
            # have the same shape. This is audit fix #2/#12: the Brain
            # Trail prototype declared tying was intended but never
            # actually shared storage.
            self.lm_head.weight = self.token_embedding.embedding.weight

        # RoPE table lives here, ONE per model, shared by every block's
        # attention -- not recomputed per-block (audit fix #15).
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
            scaling_type=getattr(config, "rope_scaling_type", None),
            scaling_factor=getattr(config, "rope_scaling_factor", 1.0),
        )

        # GPT-2 style learned absolute positions (optional)
        pos_type = (getattr(config, "position_embedding_type", "rope") or "rope").lower()
        self.position_embedding_type = pos_type
        if pos_type in ("learned", "rope_and_learned"):
            self.pos_embedding = LearnedPositionalEmbedding(
                config.max_position_embeddings, config.hidden_size
            )
        else:
            self.pos_embedding = None


        init_weights(self, num_layers=config.num_layers)
        if config.tie_word_embeddings:
            # init_weights ran on both "views" of the shared tensor above;
            # re-tie afterward in case any init step replaced the tensor
            # object rather than mutating it in place.
            self.lm_head.weight = self.token_embedding.embedding.weight

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # nn.Module.to() doesn't know about self.rope since it's not an
        # nn.Module/Parameter -- move its plain tensors manually so
        # model.to(device) actually moves the whole model, not "everything
        # except the one part with no learnable parameters."
        device = self.token_embedding.embedding.weight.device
        dtype = self.token_embedding.embedding.weight.dtype
        self.rope.to(device, dtype)
        return result

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        input_ids: (batch, seq_len) -- the whole prompt on prefill, or
                   exactly the new token(s) when kv_cache is provided
        labels:    (batch, seq_len), same shape as input_ids, for computing
                   next-token-prediction loss. Pass None for pure inference.
        attention_mask: (batch, seq_len) of 1=real token / 0=padding, for a
                   right-padded training batch. Only valid when kv_cache is
                   None -- see CausalSelfAttention's forward() for why
                   cached batched decode isn't supported yet.
        """
        x = self.token_embedding(input_ids)
        # Residual stream starts here: token (+ optional learned position) embeddings
        if getattr(self, "pos_embedding", None) is not None:
            offset = kv_cache.get_seq_length(0) if kv_cache is not None else 0
            pos = self.pos_embedding(x.size(1), offset=offset, device=x.device)
            x = x + pos.unsqueeze(0)
        # When position_embedding_type is pure "learned", still pass rope=None-safe:
        # blocks accept rope; RoPE no-ops only if attention skips — we still pass rope
        # unless type is learned-only (attention still uses RoPE if present unless disabled).
        rope = None if getattr(self, "position_embedding_type", "rope") == "learned" else self.rope
        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x, rope, kv_cache=kv_cache, layer_idx=layer_idx,
                attention_mask=attention_mask,
            )
        x = self.final_norm(x)   # Final LayerNorm / RMSNorm stabilizer
        logits = self.lm_head(x)  # Language model head -> vocab logits (Softmax outside / CE)

        loss = None
        if labels is not None:
            # Standard next-token shift: logits at position i predict the
            # token at position i+1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation. With use_cache=True (default), this
        is the real incremental-KV-cache path -- prefill once, then one
        forward pass per new token instead of recomputing the whole
        sequence every step (fixing the CRITICAL/HIGH problems in the
        audit: the Brain Trail prototype's generation always recomputed
        the full sequence, and its attempted cache path crashed outright).
        """
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()

        kv_cache = KVCache(num_layers=self.config.num_layers) if use_cache else None

        # Prefill: feed the whole prompt once, either into the cache or not.
        if use_cache:
            out = self.forward(generated, kv_cache=kv_cache)
        else:
            out = self.forward(generated)
        next_logits = out["logits"][:, -1, :]

        for _ in range(max_new_tokens):
            next_token = self._sample(next_logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            if use_cache:
                out = self.forward(next_token, kv_cache=kv_cache)
            else:
                out = self.forward(generated)
            next_logits = out["logits"][:, -1, :]

        return generated

    @staticmethod
    def _sample(
        logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            sorted_mask = cumulative - probs > top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ------------------------------------------------------------------
    # Parameter counting (Step 13: calculate, never just claim)
    # ------------------------------------------------------------------
    def count_parameters(self) -> dict[str, int]:
        def n(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        embedding_params = n(self.token_embedding)
        attn_params = sum(n(b.attention) for b in self.blocks)
        mlp_params = sum(n(b.mlp) for b in self.blocks)
        norm_params = sum(n(b.input_norm) + n(b.post_attention_norm) for b in self.blocks) + n(self.final_norm)
        lm_head_params = 0 if self.config.tie_word_embeddings else n(self.lm_head)
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "embedding": embedding_params,
            "attention": attn_params,
            "mlp": mlp_params,
            "normalization": norm_params,
            "lm_head": lm_head_params,
            "total": total,
            "trainable": trainable,
        }

    # ------------------------------------------------------------------
    # Checkpointing (audit fix #3: bundle config + metadata, not bare state_dict)
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        path: str | Path,
        tokenizer_path: str | None = None,
        training_metadata: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import brain  # local import: avoids a package-level circular import

        cfg = self.config
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": dataclasses.asdict(self.config),
                "tokenizer_path": tokenizer_path,
                "training_metadata": training_metadata or {},
                # Codebase revision that wrote this checkpoint (see
                # brain/__init__.py) -- NOT the architecture identity, which
                # is config["name"]/"architecture" below. A previous version
                # of this field just duplicated config.name under a
                # different key; that's redundant since config["name"] is
                # already saved, and it means nothing here ever actually
                # recorded which CODE wrote the checkpoint.
                "brain_version": brain.__version__,
                "architecture": cfg.name,
                # Read from self.rope (the single source of truth for the
                # NTK formula, in rope.py) instead of recomputing it here --
                # a duplicated copy of the same formula would silently go
                # stale the moment rope.py's formula is ever revised.
                "rope_theta_configured": cfg.rope_theta,
                "rope_theta_effective": float(self.rope.effective_theta),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str = "cpu") -> "BrainForCausalLM":
        path = Path(path)
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = BrainConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = BrainConfig(
        vocab_size=256, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
        intermediate_size=64, max_position_embeddings=64,
    )
    model = BrainForCausalLM(cfg)
    model.eval()

    print("=== Parameter count ===")
    counts = model.count_parameters()
    for k, v in counts.items():
        print(f"  {k}: {v:,}")

    print("\n=== Weight tying check ===")
    same_storage = model.lm_head.weight.data_ptr() == model.token_embedding.embedding.weight.data_ptr()
    print(f"lm_head and embedding share the SAME tensor storage: {same_storage}")

    print("\n=== Forward pass ===")
    input_ids = torch.randint(0, cfg.vocab_size, (2, 10))
    out = model(input_ids)
    print(f"Logits shape: {tuple(out['logits'].shape)} (expected (2, 10, {cfg.vocab_size}))")

    print("\n=== Loss + backward ===")
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    print(f"Loss: {out['loss'].item():.4f}")
    out["loss"].backward()
    grad_exists = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"All trainable params received gradients: {grad_exists}")
    model.zero_grad()

    print("\n=== Checkpoint save/load ===")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "brain_test.pt"
        model.save_checkpoint(ckpt_path, tokenizer_path="data/brain_tokenizer.json")
        reloaded = BrainForCausalLM.load_checkpoint(ckpt_path)
        reloaded.eval()
        with torch.no_grad():
            out1 = model(input_ids)["logits"]
            out2 = reloaded(input_ids)["logits"]
        print(f"Save -> load -> same output: {torch.allclose(out1, out2, atol=1e-6)}")
        print(f"Reloaded config matches: {reloaded.config == cfg}")

    print("\n=== Generation (greedy, no cache) ===")
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    out_nocache = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=False)
    print(f"Generated shape: {tuple(out_nocache.shape)} (expected (1, 11))")

    print("\n=== Generation (greedy, WITH KV cache) — must match no-cache exactly ===")
    torch.manual_seed(0)
    out_cache = model.generate(prompt, max_new_tokens=8, temperature=0.0, use_cache=True)
    match = torch.equal(out_nocache, out_cache)
    print(f"Cached generation token-for-token matches uncached: {match}")
    if not match:
        print(f"  no-cache: {out_nocache.tolist()}")
        print(f"  cache:    {out_cache.tolist()}")
