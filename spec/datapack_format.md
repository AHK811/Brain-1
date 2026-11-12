# Packed shard format (`packed-uint16-le-v1`)

Contract for the token stream that training reads. Two implementations must agree byte for byte:
`python/brainlab/data/{corpus,pack}.py` and `rust/crates/datapack`. `tests/parity/test_datapack_parity.py`
enforces it. Nothing here depends on read order, RNG, or thread count.

## Directory layout

```
<data-dir>/
  train/000000.bin 000000.json 000001.bin 000001.json ...
  val/000000.bin   000000.json
  prepare_report.json
```

`NAME.bin` is a flat array of `n_tokens` little-endian `uint16` values, row-major, no header, no padding,
no trailing newline. `NAME.json` describes that one file. `n_tokens * 2` is the exact byte length of the
`.bin`; a reader that finds extra bytes has a corrupt shard, not a format variation.

## `NAME.json`

Exactly these keys, compared after parsing (the two writers differ in key order and indentation, so the
files themselves are not byte-compared):

| key | value |
|---|---|
| `n_tokens` | element count of the `.bin` |
| `seq_len` | sequences are `n_tokens / seq_len` non-overlapping windows; `n_tokens % seq_len == 0` |
| `dtype` | `"uint16"` |
| `split` | `"train"` or `"val"` |
| `byte_order` | `"little"` |
| `row_major` | `true` |
| `source_files` | corpus-relative POSIX paths that fed this split, in the order they were packed |
| `sha256` | hex digest of the `.bin` bytes |

A sequence at index `i` covers `[i*seq_len, (i+1)*seq_len)`; the target for that window is the same
range shifted by one. Because the last token of the final window in a shard has no lookahead inside that
shard, a reader may gather across shard boundaries; it must never drop a window to avoid doing so.

## Document splitting

A corpus file is split into documents at `\n` + one or more ASCII whitespace (`space \t \n \r \v \f`) +
`\n`, taking the **last** newline of the run as the boundary. Each document is trimmed of leading and
trailing whitespace; empty documents are dropped. All characters inside one document are kept verbatim.

## File-level train/val split

Corpus files are sorted by their POSIX-relative path (`a/b.txt` before `a/c.txt`, byte comparison). With
`stride = max(1, floor(1/val_ratio + 0.5))`, file at index `i` goes to **val** when `i % stride == 0`,
otherwise **train**. `val_ratio = 0` means no val split at all. This is deterministic and independent of
file contents, so re-running after adding one file only changes splits from that index onward; it also
needs no shared RNG implementation between Python and Rust, which is the reason a seeded shuffle was
rejected.

## Packing

Documents are encoded in file order, then within a file in document order. Each document contributes its
token ids followed by exactly one `eos_id` (the separator is the end-of-document token, not a learned
token). The concatenated stream is truncated to a multiple of `seq_len` - the tail is discarded, never
zero-padded, because padding would teach the model a token that does not exist in the vocab's semantic
range. The stream is then cut into chunks of `shard_tokens` (last chunk may be short; chunks shorter
than `seq_len` are dropped).

`uint16` caps ids at 65535, so `vocab_size <= 65536` is enforced by the packer, not by convention.

## Cross-language entry points

| operation | Python | Rust |
|---|---|---|
| document split | `data.corpus.documents_from_text` | `datapack::split_documents` |
| file split | `data.corpus.split_files` | `datapack::split_by_file` |
| stride | `data.corpus.val_stride` | `datapack::val_stride` |
| pack stream | `data.pack.pack_stream` | `datapack::pack_stream` |
| write shards | `data.pack.write_shards` | `datapack::write_shards` |
| CLI | `python -m brainlab.data.prepare` | `cargo run -p brainlab-datapack --bin pack` |

## The committed fixture

`tests/golden/datapack_fixture/` is the byte-level arbiter:

```
corpus/group-*/file-*.txt      5 files, from the tokenizer fixture documents
tokenizer/tokenizer.bin        the 272-token BLTK vocab from tests/golden/tokenizer_parity.json
rust/train/*.bin|.json         written by the RUST packer
rust/report.json               what the rust CLI claimed it wrote
```

Config is `seq_len=16`, `shard_tokens=48`, `val_ratio=0.25` (stride 4, so files 0 and 4 are val). Python
recomputes the same stream from `corpus/` and must match the Rust `.bin` byte for byte; the manifests are
compared field by field. Regenerate the Rust side with `just datapack-fixture`; the inputs themselves come
from `python -m brainlab.tok.parity_fixture --datapack tests/golden/datapack_fixture`.

