# Tokenizer spec

Byte-level BPE with a **flat id space**: ids `0..255` are single bytes, ids `256..vocab-specials-1` are learned merges, and the final ids are reserved specials. Rust trains and encodes; `python/brainlab/tok/reference.py` is a second, independent implementation of this same spec, and `tests/parity/test_tokenizer_parity.py` requires **identical id sequences on every input**. That parity test is the point of this file: two implementations of one written contract, not one implementation copied twice.

## Id space

| range | meaning |
|---|---|
| `0 .. 255` | the single UTF-8 byte with that value |
| `256 .. 32753` | learned merges, assigned in the order they are learned |
| `32754` | `<\|image_start\|>` |
| `32755` | `<\|image_end\|>` |
| `32756` | `<\|thought_start\|>` |
| `32757` | `<\|thought_end\|>` |
| `32758` | `<\|call_start\|>` |
| `32759` | `<\|call_end\|>` |
| `32760` | `<\|response_start\|>` |
| `32761` | `<\|response_end\|>` |
| `32762` | `<\|system\|>` |
| `32763` | `<\|user\|>` |
| `32764` | `<\|assistant\|>` |
| `32765` | `<\|turn_end\|>` |
| `32766` | `<pad>` |
| `32767` | `<eos>` |

The concrete ids in the table are for `vocab_size = 32768`; the layout generalizes to any vocab: the 14 specials are always the last 14 ids in exactly the order above, `<pad>` second-to-last and `<eos>` last. Format version 1 (legacy, 2 specials only) keeps just `<pad>` and `<eos>` at the tail; everything else in this file applies to both.

Reserving the **top** of the range (not the bottom) keeps `id -> byte` identity for the first 256 ids, which makes encode/decode trivially reversible and keeps every id inside `uint16`. BPE training stops at `vocab_size - 14` merges. There is no `<unk>`: bytes cover any input, so unknown-token handling does not exist (`emit_unk = false`).

## Control tokens

The 12 control tokens are **atomic and protected**:

- Merges are never trained whose byte string contains any special's spelling, so a control token can never appear inside an ordinary merge token (`merge {rank} is too long or contains protected bytes`).
- Encoding only emits a control id when the encoder is explicitly told specials are allowed (`allow_special = true`); otherwise the spelling is encoded as ordinary text. This keeps untrusted user input inert: a prompt containing the literal text `<|assistant|>` cannot inject the assistant role.
- The chat format uses `<|system|>`, `<|user|>`, `<|assistant|>` as turn roles, `<|turn_end|>` to close each turn, `<|thought_start|>` / `<|thought_end|>` for weighted reasoning spans, and `<|call_start|>` / `<|call_end|>` plus `<|response_start|>` / `<|response_end|>` for the interleaved tool loop. `<|image_start|>` / `<|image_end|>` are reserved for future multimodal frames.

## Pre-tokenizer: `whitespace-plus-punctuation-v1`

Applied to the whole file as a Unicode code-point sequence. Merges never cross a word boundary.

Scan left to right, emitting maximal words by the first matching rule:

1. **Whitespace run** - one or more code points where `unicode whitespace` is true. The whole run is one word.
2. **Other run** - one or more code points that are neither alphabetic/numeric (per Unicode general category `L*` or `N*`) nor whitespace, and are not `-`. The whole run is one word.
3. **Alphanumeric run** - maximal run of `L*`/`N*` plus a single `-` only when it sits between two alphanumeric code points (`state-of-the-art` is one word).
   - If every code point in the run is an ASCII digit `0`-`9` and the run is longer than 3 code points, split from the **right** into groups of 3: `1234567` -> `1`, `234`, `567`. Unicode digits outside ASCII (Arabic-Indic, fullwidth) are treated as letters: the `Nd` category tables differ between runtimes, and agreeing matters more than covering them.

Each word is encoded to UTF-8 bytes; that byte string is the unit BPE operates on. A word whose UTF-8 length exceeds `max_word_bytes = 32` is split into consecutive 32-byte chunks, cutting only at UTF-8 code-point boundaries.

This is deliberately **not** the GPT-2 regex. GPT-2 remaps byte values `33..126` and `161..172` into a private Unicode range to make whitespace visible in the vocab; that remapping is a display trick that costs cross-language exactness. Our vocab stores raw bytes. Do not compare token ids against `GPT2TokenizerFast`.

## Training

Input: every word emitted by the pre-tokenizer over the corpus. Word **types** are the distinct byte strings, indexed in **lexicographic byte order**, each with its frequency. Indexing by the word's bytes rather than by where it first appears means the vocab does not depend on the order files are read in - the same corpus always produces the same `tokenizer.bin`, byte for byte.

```
words  = pre_tokenizer(text)              # ordered
freqs  = count identical byte-strings
syms[t] = list(byte) for each word type   # ids start as raw byte values
pairs  = all adjacent (a,b) across all word types with their counts

repeat until vocab_size - 14 merges learned or no pair occurs more than once:
    best = argmax over pairs of (count, -first_index, -rank_a, -rank_b)
           where first_index = smallest word-type index at which the pair occurs
    new_id = next id (256, 257, ...)
    for every word type containing (a,b):
        replace all non-overlapping left-to-right occurrences of (a,b) with new_id
        update its count proportionally
    recompute first_index for surviving pairs
```

The tie-break is `(-count, first_index, rank_a, rank_b)` and it must be applied exactly: two different tie-break rules produce two different vocabs, and the parity test then fails for reasons that have nothing to do with a bug in either implementation. If the loop cannot make progress (best count `<= 1`), stop and emit a vocab smaller than requested; `vocab_size` is an upper bound.

The byte-string of `new_id` is `bytes(a) ++ bytes(b)`, so decoding any id list is `b"".join(token_bytes[i] for i in ids)` then `utf-8 decode(errors="replace")`.

## Files

### `vocab.json`
`{"<token string as latin-1-escaped text>": id, ...}` for all ids. Keys for byte tokens are the raw byte rendered as a `\u00XX` escape; keys for merge tokens and specials are their byte string under the same rule. Written with `ensure_ascii = true`, 2-space indent, keys sorted by id ascending.

### `merges.txt`
Line 1 is exactly `#version: 1.0` for the legacy 2-special layout or `#version: 2.0` for the 14-special layout. Then one line per merge in learned order: `id_a id_b` (decimal, space separated, `\n` terminated, no trailing blank line). Numeric merges are the only form; string-form merges are not used.

### `tokenizer.bin`
Little-endian, no padding:

```
0  u8[4]  magic "BLTK"
4  u16    format version = 2
6  u32    vocab_size
10 u32    n_merges
14 u8[]   for id in 0..vocab_size-1:  u8 len, then len bytes
          then n_merges * (u32 a, u32 b)
          then u32 eos_id, u32 pad_id
```

`len` is at most 32 (the table stores tokens as length-prefixed bodies; the header's `len` byte caps token bodies at 255). `tokenizer.bin` is the only file inference loaders read; the JSON/text pair exists so humans and Hugging Face tooling can inspect the vocab.

## Encode contract

```
for word in pre_tokenizer(text):
    syms = list(word.encode(utf-8))                  # ids 0..255
    loop:
        i = index of leftmost adjacent pair whose rank exists in merges
        if none: break
        merge that single leftmost occurrence
    emit syms
```

One merge per pass, leftmost first, re-scan after each merge. Minimum rank wins; when the same minimum rank appears at several positions the leftmost is merged, and the next pass may merge the next occurrence. Merge passes never cross word boundaries, so an id sequence is the concatenation of per-word results and whitespace is preserved inside tokens. With `allow_special`, control spellings are recognized greedily left-to-right and emitted as single ids before the surrounding text is encoded; without it they are ordinary text and can only be encoded through ordinary tokens.

## Decode contract

`ids -> bytes` by concatenating `token_bytes[id]`, then UTF-8 decode with `errors="replace"`. Decoding the id sequence produced by encoding the same text must return it byte-for-byte for all valid UTF-8 input; the parity test asserts `decode(encode(t)) == t` on the full corpus.

## Packing contract

`rust/crates/datapack` turns text into shards under `data/packed/{train,val}/`:

- `NNNNNN.bin` - `uint16` little-endian token stream, length an exact multiple of `seq_len`.
- `NNNNNN.json` - `{"n_tokens", "seq_len", "dtype": "uint16le", "split", "source_files"}`.

Sequences are packed back-to-back with a single `<eos>` (`32767`) between documents, no padding inside a sequence, no document split across a shard boundary unless it is longer than `seq_len`. Split into train/val by file, never by sequence, using `seed` from `spec/model_config.toml` to choose the val files - a document that appears in train and val inflates the eval number, and a 105M model will happily memorize it.
