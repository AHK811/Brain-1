from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from brainlab.data.corpus import documents_from_text, read_text_dir, split_files, val_stride
from brainlab.data.pack import PackConfig, pack_stream, split_shards, write_shards
from brainlab.data.shards import TokenShards
from brainlab.spec.config import repo_root
from brainlab.tok.reference import Tokenizer

FIXTURE_DIR = repo_root() / "tests" / "golden" / "datapack_fixture"
RUST_PACKED = os.environ.get("BRAINLAB_RUST_PACKED_DIR", str(FIXTURE_DIR / "rust"))
CORPUS = {
    "a/first.txt": "The model trains.\n\nThe gradient flows through every branch.\n\n\n尾部\n",
    "a/second.txt": "attention costs grow with the square of the length\nand causal masking matters\n",
    "b/third.txt": "one document only\n",
    "b/fourth.txt": "p\n\nq r s\n\nt u v w x y z\n",
    "fifth.txt": "\n\n\n",
}


def test_val_stride_rounds_half_up_and_matches_the_spec() -> None:
    assert val_stride(0.1) == 10
    assert val_stride(0.005) == 200
    assert val_stride(0.25) == 4
    assert val_stride(1 / 3) == 3
    assert val_stride(0.08) == 13
    assert val_stride(0.0) == (1 << 64) - 1


def test_file_split_is_deterministic_and_disjoint() -> None:
    items = sorted(CORPUS.items())
    train, val = split_files(items, 0.25)
    again = split_files(sorted(reversed(items)), 0.25)
    assert train == again[0] and val == again[1]
    assert not (set(name for name, _ in train) & set(name for name, _ in val))
    assert len(train) + len(val) == len(items)
    assert [name for name, _ in val] == ["a/first.txt", "fifth.txt"]
    assert split_files(items, 0.0)[1] == []


def test_document_split_uses_the_last_newline_of_a_whitespace_run() -> None:
    assert documents_from_text("a\nb\nc") == ["a\nb\nc"]
    assert documents_from_text("a\n\nb") == ["a", "b"]
    assert documents_from_text("a\n \t\nb") == ["a", "b"]
    assert documents_from_text("a\n\n \n\nb") == ["a", "b"]
    assert documents_from_text("\n\n   \n") == []


def test_pack_appends_one_eos_per_document_and_truncates_the_tail() -> None:
    cfg = PackConfig(seq_len=4, shard_tokens=8, eos_id=257)
    stream = pack_stream([[1, 2, 3], [4, 5]], cfg)
    assert stream.tolist() == [1, 2, 3, 257]
    assert stream.size % cfg.seq_len == 0
    long = pack_stream([[7] * 9], cfg)
    assert long.size == 8
    assert int(long[-1]) != 257


def test_shard_files_are_raw_little_endian_uint16_with_a_matching_digest(tmp_path: Path) -> None:
    import hashlib

    cfg = PackConfig(seq_len=4, shard_tokens=8, eos_id=257)
    docs = [[index % 256 for index in range(30)] for _ in range(3)]
    stream = pack_stream(docs, cfg)
    written = write_shards(split_shards(stream, cfg), tmp_path, "train", cfg, ["a/first.txt"])
    assert len(written) == 12
    for offset, path in enumerate(written):
        raw = path.read_bytes()
        manifest = json.loads((path.with_suffix(".json")).read_text(encoding="utf-8"))
        assert len(raw) == manifest["n_tokens"] * 2
        assert manifest["n_tokens"] % cfg.seq_len == 0
        assert manifest["dtype"] == "uint16"
        assert manifest["byte_order"] == "little"
        assert manifest["source_files"] == ["a/first.txt"]
        assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
        values = np.frombuffer(raw, dtype="<u2")
        begin = offset * cfg.shard_tokens
        assert values.tolist() == stream[begin : begin + int(manifest["n_tokens"])].tolist()


def test_reader_gathers_sequences_across_shard_boundaries(tmp_path: Path) -> None:
    cfg = PackConfig(seq_len=4, shard_tokens=8, eos_id=257)
    docs = [[index % 256 for index in range(30)] for _ in range(3)]
    stream = pack_stream(docs, cfg)
    write_shards(split_shards(stream, cfg), tmp_path, "train", cfg)
    reader = TokenShards(tmp_path, seq_len=4)
    assert len(reader) == stream.size // 4 - 1
    first, target = reader[1]
    assert first.tolist() == stream[4:8].tolist()
    assert target.tolist() == stream[5:9].tolist()
    last_input, last_target = reader[len(reader) - 1]
    assert last_input.shape == (4,) and last_target.shape == (4,)
    assert reader.total_tokens == stream.size


@pytest.mark.skipif(
    not (FIXTURE_DIR / "rust" / "train").is_dir(),
    reason="run `just datapack-fixture` to produce the rust shards",
)
def test_rust_packer_writes_identical_shards() -> None:
    import hashlib

    files = read_text_dir(FIXTURE_DIR / "corpus")
    tokenizer = Tokenizer.load(FIXTURE_DIR / "tokenizer")
    cfg = PackConfig(seq_len=16, shard_tokens=48, eos_id=tokenizer.eos_id)
    train, val = split_files(files, 0.25)
    covered = False
    for name, items in (("train", train), ("val", val)):
        docs = [document for _, text in items for document in documents_from_text(text)]
        stream = pack_stream([tokenizer.encode(document) for document in docs], cfg)
        shard_dir = FIXTURE_DIR / "rust" / name
        paths = sorted(shard_dir.glob("*.bin"))
        assert paths, f"{name}: the rust packer wrote no shards"
        for index, path in enumerate(paths):
            covered = True
            produced = np.frombuffer(path.read_bytes(), dtype="<u2")
            begin = index * cfg.shard_tokens
            assert produced.tolist() == stream[begin : begin + produced.size].tolist(), f"{name} shard {index} differs"
            manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            assert manifest["n_tokens"] == produced.size
            assert manifest["seq_len"] == cfg.seq_len
            assert manifest["split"] == name
            assert manifest["source_files"] == [item[0] for item in items]
            assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        mine = json.loads((shard_dir / "000000.json").read_text(encoding="utf-8"))
        assert mine["dtype"] == "uint16" and mine["byte_order"] == "little" and mine["row_major"] is True
    assert covered, "the fixture produced no shards to compare"
