from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from brainlab.data.pack import PackConfig, pack_and_write
from brainlab.data.shards import ShardBatcher, TokenShards
from brainlab.train.optimizer import peak_lrs, scale_lr
from brainlab.train.train import cycle, resume_cursor

CFG = PackConfig(seq_len=32, shard_tokens=32 * 8, eos_id=1)


def make_shards(directory: Path) -> TokenShards:
    docs = [[(index * 7 + step) % 997 + 2 for step in range(32)] for index in range(24)]
    pack_and_write(docs, directory, "train", CFG)
    return TokenShards(directory, 32)


def first_tokens(batches) -> list[int]:
    return [int(inputs[0, 0]) for inputs, _ in batches]


def test_resume_replays_the_same_batches_as_an_uninterrupted_run(tmp_path: Path) -> None:
    # one directory per batcher: TokenShards keeps its memmaps open, and Windows will not
    # let a second pack truncate a file that is still mapped.
    shards = make_shards(tmp_path / "a")
    per_epoch = len(ShardBatcher(shards, 2, shuffle_seed=11))
    assert per_epoch >= 4, "the fixture is too small to cross an epoch boundary"
    uninterrupted = ShardBatcher(make_shards(tmp_path / "b"), 2, shuffle_seed=11)
    straight = first_tokens(islice(cycle(uninterrupted), per_epoch + 3))

    fresh = ShardBatcher(make_shards(tmp_path / "c"), 2, shuffle_seed=11)
    resumed = first_tokens(islice(cycle(fresh, per_epoch), 3))
    assert resumed == straight[per_epoch : per_epoch + 3], "a resumed run must not re-serve batches it already trained on"
    assert fresh.epoch == 1, "the cursor should land one epoch in, not restart the shuffle"


def test_resume_across_several_epochs(tmp_path: Path) -> None:
    per_epoch = len(ShardBatcher(make_shards(tmp_path / "a"), 2, shuffle_seed=5))
    uninterrupted = ShardBatcher(make_shards(tmp_path / "b"), 2, shuffle_seed=5)
    straight = first_tokens(islice(cycle(uninterrupted), 2 * per_epoch + 2))
    fresh = ShardBatcher(make_shards(tmp_path / "c"), 2, shuffle_seed=5)
    assert first_tokens(islice(cycle(fresh, 2 * per_epoch), 2)) == straight[-2:]


def test_cursor_arithmetic() -> None:
    assert resume_cursor(0, 10) == (0, 0)
    assert resume_cursor(9, 10) == (0, 9)
    assert resume_cursor(10, 10) == (1, 0)
    assert resume_cursor(37, 10) == (3, 7)
    for bad in ((-1, 10), (5, 0)):
        with pytest.raises(ValueError, match="cannot place batch"):
            resume_cursor(*bad)


def test_start_batch_wider_than_an_epoch_wraps_instead_of_rejecting(tmp_path: Path) -> None:
    per_epoch = len(ShardBatcher(make_shards(tmp_path / "a"), 2, shuffle_seed=3))
    uninterrupted = ShardBatcher(make_shards(tmp_path / "b"), 2, shuffle_seed=3)
    expected = first_tokens(islice(cycle(uninterrupted), 4 * per_epoch, 4 * per_epoch + 2))
    fresh = ShardBatcher(make_shards(tmp_path / "c"), 2, shuffle_seed=3)
    assert first_tokens(islice(cycle(fresh, 4 * per_epoch), 2)) == expected
    assert fresh.epoch == 4, "the resumed batcher sits inside epoch 4, it has not finished it"


def test_a_changed_lr_config_beats_the_lr_saved_in_the_checkpoint() -> None:
    param = torch.zeros(4, requires_grad=True)
    optim = AdamW([param], lr=0.04)
    peaks = peak_lrs([optim])
    optim.load_state_dict({"param_groups": [{"lr": 0.02, "params": [0]}], "state": {}})
    assert optim.param_groups[0]["lr"] == 0.02, "the checkpoint carries the old schedule"
    applied = scale_lr([optim], 1.0, peaks)
    assert optim.param_groups[0]["lr"] == 0.04, "resume must follow the config, not the saved peak"
    assert applied == {"optim0": 0.04}


def test_scale_lr_schedules_from_the_peak_not_the_current_value() -> None:
    optim = AdamW(
        [
            {"params": [torch.zeros(4, requires_grad=True)], "lr": 0.02},
            {"params": [torch.zeros(4, requires_grad=True)], "lr": 0.01},
        ]
    )
    peaks = peak_lrs([optim])
    scale_lr([optim], 0.5, peaks)
    scale_lr([optim], 0.25, peaks)
    assert [group["lr"] for group in optim.param_groups] == [0.005, 0.0025]


def test_scale_lr_rejects_a_mismatched_schedule() -> None:
    optim = AdamW([torch.zeros(4, requires_grad=True)], lr=0.02)
    with pytest.raises(ValueError, match="peak schedules"):
        scale_lr([optim, optim], 1.0, peak_lrs([optim]))
    with pytest.raises(ValueError, match="non-negative"):
        scale_lr([optim], -1.0, peak_lrs([optim]))
