"""
scripts/tiny_overfit_test.py

Step 16's tiny overfit test: train on 3 hardcoded sentences until loss
drops and the model can reproduce them via greedy decoding. This is NOT
the real pretraining pipeline (that's Phase 3, explicitly deferred) -- it
is an architectural correctness test that the whole forward/backward/
optimizer stack actually learns something, not just runs without crashing.

Uses a SMALL model (not the full 27M brain_0_2_30m config) so this runs in
seconds on CPU -- the point is to prove the training mechanics are wired
correctly, not to produce a useful model. Full-scale training is Phase 3.

Run with: PYTHONPATH=. python3 scripts/tiny_overfit_test.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from brain.core.config import BrainConfig
from brain.model.brain_model import BrainForCausalLM
from brain.tokenizer.tokenizer import BrainTokenizer

SENTENCES = [
    "The brain is learning.",
    "Brain can process text.",
    "Braegn builds AI systems.",
]


def main() -> int:
    torch.manual_seed(0)

    print("Training a throwaway tokenizer on the 3 overfit sentences...")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "overfit_corpus.txt"
        corpus_path.write_text("\n".join(SENTENCES))
        tok = BrainTokenizer.train(str(corpus_path), vocab_size=256, min_frequency=1)

    batch = tok.encode_batch(SENTENCES, pad=True, add_special_tokens=True)
    input_ids = torch.tensor(batch["input_ids"])
    print(f"Tokenized batch shape: {tuple(input_ids.shape)}")

    # Small config -- big enough to have real capacity, small enough to be fast.
    config = BrainConfig(
        vocab_size=tok.vocab_size,
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=int(64 * 8 / 3),
        max_position_embeddings=64,
        dropout=0.0,
    )
    model = BrainForCausalLM(config)
    model.train()
    n_params = model.count_parameters()["total"]
    print(f"Model: {n_params:,} parameters (small test config, NOT brain_0_2_30m)")

    # Labels: mask PAD positions out of the loss with -100, since cross_entropy
    # treats -100 as ignore_index (see brain_model.py's forward()).
    labels = input_ids.clone()
    labels[labels == tok.pad_id] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    print("\nTraining...")
    losses = []
    n_steps = 300
    for step in range(n_steps):
        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    print(f"\nInitial loss (avg of first 5 steps): {initial_loss:.4f}")
    print(f"Final loss   (avg of last 5 steps):  {final_loss:.4f}")
    loss_decreased = final_loss < initial_loss * 0.3
    print(f"Loss decreased substantially (>70% reduction): {loss_decreased}")

    print("\nGreedy continuation from each sentence's own prefix (not bare <bos>):")
    print("(Generating unconditionally from <bos> alone can't distinguish between")
    print(" 3 memorized sentences under greedy/deterministic decoding -- that would")
    print(" test the wrong thing. Prompting with each sentence's distinguishing")
    print(" prefix and checking the completion is the correct overfit check.)\n")
    model.eval()
    correct = 0
    all_encoded = [tok.encode(s, add_special_tokens=True) for s in SENTENCES]

    def shortest_unique_prefix_len(idx: int) -> int:
        """Shortest prefix length (in tokens) that no other sentence shares
        -- 'Brain' and 'Braegn' share initial byte-level tokens, so a fixed
        prefix length isn't guaranteed to disambiguate all pairs."""
        target = all_encoded[idx]
        for length in range(2, len(target) + 1):
            candidate = target[:length]
            if not any(
                i != idx and all_encoded[i][:length] == candidate
                for i in range(len(all_encoded))
            ):
                return length
        return len(target)

    for i, sentence in enumerate(SENTENCES):
        full_ids = all_encoded[i]  # [bos, ..., eos]
        prefix_len = shortest_unique_prefix_len(i) + 1  # +1 token of margin
        prompt_ids = torch.tensor([full_ids[:prefix_len]])
        with torch.no_grad():
            gen = model.generate(
                prompt_ids, max_new_tokens=20, temperature=0.0,
                eos_token_id=tok.eos_id, use_cache=True,
            )
        gen_text = tok.decode(gen[0].tolist())
        match = gen_text.strip() == sentence.strip()
        correct += match
        print(f"  target:    {sentence!r}")
        print(f"  prompt:    {tok.decode(full_ids[:prefix_len])!r}")
        print(f"  generated: {gen_text!r}")
        print(f"  exact match: {match}\n")

    print(f"Sentences exactly reproduced: {correct}/{len(SENTENCES)}")

    success = loss_decreased and correct >= 2  # allow one near-miss, still proves learning
    print(f"\n{'PASS' if success else 'FAIL'}: tiny overfit test "
          f"{'demonstrates the training pipeline learns correctly' if success else 'DID NOT show expected learning -- investigate before proceeding'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
