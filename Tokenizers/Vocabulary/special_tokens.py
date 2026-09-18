from __future__ import annotations
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
CORE = [PAD, UNK, BOS, EOS]
CHAT = ["<|system|>", "<|user|>", "<|assistant|>", "<|think|>", "<|end_think|>"]
TOOL = ["<|tool_call|>", "<|end_tool_call|>", "<|tool_result|>"]
MM = ["<|image_start|>", "<|image_end|>", "<|image_pad|>", "<|audio_start|>", "<|audio_end|>"]
ALL_SPECIAL = CORE + CHAT + TOOL + MM
