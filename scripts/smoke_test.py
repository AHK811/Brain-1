"""
scripts/smoke_test.py

The exact checklist from the implementation prompt's Step 15, run end to
end against the REAL brain_0_2_30m.yaml config (not a toy config) and the
tokenizer trained on data/sample_corpus.txt. Every line below prints a
pass/fail rather than assuming success -- "report any failures honestly"
is a direct instruction, not just a nicety.

Run with:  PYTHONPATH=. python3 scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from brain.core.config import BrainConfig
from brain.model.brain_model import BrainForCausalLM
from brain.tokenizer.tokenizer import BrainTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = str(REPO_ROOT / "data" / "sample_corpus.txt")
CONFIG_PATH = str(REPO_ROOT / "configs" / "model" / "brain_0_2_30m.yaml")

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    torch.manual_seed(0)

    print("=" * 70)
    print("Brain v0.2 — Smoke Test (Step 15 checklist, run against real config)")
    print("=" * 70)

    # --- Tokenizer ---------------------------------------------------------
    print("\n[Tokenizer]")
    tok = BrainTokenizer.train(CORPUS_PATH, vocab_size=2048, min_frequency=1)
    check("tokenizer trains", tok.vocab_size > 4)

    text = "Hello Brain"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids)
    check("tokenizer encode/decode round-trip", decoded == text, f"{text!r} -> {decoded!r}")

    with tempfile.TemporaryDirectory() as tmp:
        tok_path = Path(tmp) / "tok.json"
        tok.save(tok_path)
        reloaded_tok = BrainTokenizer.from_file(tok_path)
        check("tokenizer save/load", reloaded_tok.vocab_size == tok.vocab_size)

    # --- Model instantiation -------------------------------------------------
    print("\n[Model]")
    config = BrainConfig.from_yaml(CONFIG_PATH)
    model = BrainForCausalLM(config)
    check("model instantiates from real brain_0_2_30m.yaml", True)

    param_counts = model.count_parameters()
    check(
        "parameter count matches audit calculation (26,944,896)",
        param_counts["total"] == 26_944_896,
        f"got {param_counts['total']:,}",
    )

    # --- Forward / backward --------------------------------------------------
    print("\n[Forward / backward]")
    prompt_ids = tok.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids])
    out = model(input_ids)
    check(
        "forward pass produces correct logits shape",
        out["logits"].shape == (1, len(prompt_ids), config.vocab_size),
        f"got {tuple(out['logits'].shape)}",
    )

    out = model(input_ids, labels=input_ids.clone())
    check("loss computes and is finite", torch.isfinite(out["loss"]).item(), f"loss={out['loss'].item():.4f}")

    out["loss"].backward()
    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    check("backward pass populates gradients on all trainable params", grads_ok)
    model.zero_grad()

    # --- Checkpointing ---------------------------------------------------------
    print("\n[Checkpointing]")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "brain_v0_2.pt"
        model.save_checkpoint(ckpt_path, tokenizer_path=str(tok_path))
        check("checkpoint saves", ckpt_path.exists())

        reloaded_model = BrainForCausalLM.load_checkpoint(ckpt_path)
        model.eval()
        reloaded_model.eval()
        with torch.no_grad():
            out1 = model(input_ids)["logits"]
            out2 = reloaded_model(input_ids)["logits"]
        check("checkpoint loads with identical output", torch.allclose(out1, out2, atol=1e-6))

    # --- Generation + KV cache ----------------------------------------------
    print("\n[Generation / KV cache]")
    model.eval()
    torch.manual_seed(0)
    gen_nocache = model.generate(input_ids, max_new_tokens=10, temperature=0.0, use_cache=False)
    check("generation (no cache) runs and extends sequence", gen_nocache.shape[1] == input_ids.shape[1] + 10)

    torch.manual_seed(0)
    gen_cache = model.generate(input_ids, max_new_tokens=10, temperature=0.0, use_cache=True)
    check("generation (KV cache) runs and extends sequence", gen_cache.shape[1] == input_ids.shape[1] + 10)

    cache_matches = torch.equal(gen_nocache, gen_cache)
    check(
        "MANDATORY: cached generation token-for-token matches uncached",
        cache_matches,
        "STOP AND DEBUG per Step 10 if this fails" if not cache_matches else "",
    )

    generated_text = tok.decode(gen_cache[0].tolist())
    check("generated output decodes without crashing", isinstance(generated_text, str), repr(generated_text))

    # --- Summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        print("\nFAILURES:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
