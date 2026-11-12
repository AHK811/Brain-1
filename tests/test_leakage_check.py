from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brainlab.data.leakage import check, rolling_hashes
from brainlab.data.pack import PackConfig, pack_and_write
from tests.conftest import repo_root

CFG = PackConfig(seq_len=64, shard_tokens=64 * 8, eos_id=1)


def docs(count: int, base: int, seq_len: int = 64) -> list[list[int]]:
    return [[(base + index * 37 + step) % 2048 + 2 for step in range(seq_len)] for index in range(count)]


def test_hashes_are_order_sensitive() -> None:
    forward = rolling_hashes(np.array([1, 2, 3]), 2)
    backward = rolling_hashes(np.array([3, 2, 1]), 2)
    assert forward.size == 2
    assert not np.array_equal(forward, backward)


def test_a_copy_between_splits_is_found(tmp_path: Path) -> None:
    stolen = docs(1, 700)[0]
    pack_and_write([stolen] + docs(6, 10), tmp_path / "train", "train", CFG)
    pack_and_write([stolen] + docs(3, 900), tmp_path / "val", "val", CFG)
    report = check(tmp_path / "train", tmp_path / "val", 64, sequences=8, span=20)
    assert report["matching_windows"] >= 1, "the stolen document produced no window hits"
    assert report["leaked"]
    assert report["examples"], "a leak report with no example cannot be acted on"
    assert report["examples"][0]["tokens"][:20] == stolen[:20]


def test_disjoint_splits_are_clean(tmp_path: Path) -> None:
    pack_and_write(docs(6, 10), tmp_path / "train", "train", CFG)
    pack_and_write(docs(4, 900), tmp_path / "val", "val", CFG)
    report = check(tmp_path / "train", tmp_path / "val", 64, sequences=8, span=20)
    assert report["matching_windows"] == 0
    assert not report["leaked"]


def test_a_repetitive_corpus_matches_without_any_copy(tmp_path: Path) -> None:
    sentence = [7, 8, 9, 10, 11, 12, 13, 14] * 8
    pack_and_write([sentence] * 4, tmp_path / "train", "train", CFG)
    pack_and_write([sentence] * 2, tmp_path / "val", "val", CFG)
    report = check(tmp_path / "train", tmp_path / "val", 64, sequences=4, span=20)
    assert report["matching_windows"] > 0, "span matching must flag genuine repetition, not only file copies"


def test_span_wider_than_a_sequence_cannot_report_clean(tmp_path: Path) -> None:
    pack_and_write(docs(3, 10), tmp_path / "train", "train", CFG)
    pack_and_write(docs(2, 900), tmp_path / "val", "val", CFG)
    with pytest.raises(SystemExit, match="does not fit"):
        check(tmp_path / "train", tmp_path / "val", 64, sequences=4, span=65)


def test_committed_fixture_split_keeps_its_files_disjoint() -> None:
    root = repo_root() / "tests" / "golden" / "datapack_fixture" / "rust"
    if not (root / "train").is_dir():
        pytest.skip("the datapack fixture has not been generated")
    report = check(root / "train", root / "val", 16, sequences=40, span=16)
    assert report["shared_source_files"] == [], "the stride split must not put a file in both splits"
    assert report["val_windows"] > 0
    assert report["matching_windows"] > 0, (
        "this fixture is synthetic repetitive text: even a whole 16-token sequence repeats across "
        "documents, which is why span has to be long enough that coincidence is impossible"
    )
