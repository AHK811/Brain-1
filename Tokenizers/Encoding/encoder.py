from __future__ import annotations
from typing import Sequence
class SimpleEncoder:
    """Whitespace fallback encoder when BPE unavailable."""
    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab = vocab or {"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
        self.id_to_token = {i:t for t,i in self.vocab.items()}
        self.unk_id = self.vocab.get("<unk>", 1)
        self.pad_id = self.vocab.get("<pad>", 0)
        self.bos_id = self.vocab.get("<bos>", 2)
        self.eos_id = self.vocab.get("<eos>", 3)
    def encode(self, text: str, add_special: bool = True) -> list[int]:
        toks = text.split()
        ids = [self.vocab.get(t, self.unk_id) for t in toks]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids
    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        special = {self.pad_id, self.bos_id, self.eos_id}
        toks = []
        for i in ids:
            if skip_special and i in special: continue
            toks.append(self.id_to_token.get(i, "<unk>"))
        return " ".join(toks)
