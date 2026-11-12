from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer

from brainlab.data.shards import ShardBatcher, TokenShards
from brainlab.export.to_safetensors import load_model as load_safetensors_model
from brainlab.export.to_safetensors import save_model
from brainlab.model.gpt import TransformerLM
from brainlab.spec.config import load_model_spec, load_raw_config
from brainlab.train.checkpoint import latest_checkpoint, load_checkpoint, rotate, save_checkpoint
from brainlab.train.optimizer import build_optimizers, scale_lr
from brainlab.train.schedule import LRSchedule, tokens_per_step
from brainlab.util import configure_stdout


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="brainlab-train", description="Train the reference transformer.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/packed/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/packed/val"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default=None)
    parser.add_argument("--optimizer", choices=("muon", "adamw"), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--accum", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None, help="safetensors checkpoint to warm start from")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return device


def git_state() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "no-git"
    dirty = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True
    ).stdout.strip()
    if untracked:
        dirty = True
    return rev + ("+dirty" if dirty else "")


def cycle(batcher: ShardBatcher) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    while True:
        yield from batcher


def evaluate(
    model: TransformerLM,
    batcher: ShardBatcher,
    device: torch.device,
    max_batches: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for index, (x, y) in enumerate(batcher):
            if index >= max_batches:
                break
            ids = torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
            targets = torch.from_numpy(np.ascontiguousarray(y)).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                _, loss = model(ids, targets)
            counted = int(np.count_nonzero(y != -1))
            total_loss += float(loss.detach().float()) * counted
            total_tokens += counted
    model.train()
    if total_tokens == 0:
        raise SystemExit("evaluation had no labelled tokens")
    mean = total_loss / total_tokens
    return {"val_loss": mean, "val_ppl": math.exp(mean), "val_tokens": float(total_tokens)}


def divergence_reason(
    mean_loss: float,
    grad_norm: float,
    skipped: bool,
    consecutive_skips: int,
    zero_grad_streak: int,
    scale: float,
    amp_enabled: bool,
    muon_skips: int,
    skip_limit: int,
    zero_grad_limit: int,
    scale_floor: float,
) -> str:
    if not math.isfinite(mean_loss) and not skipped:
        return f"non-finite loss {mean_loss}"
    if not math.isfinite(grad_norm) and not skipped:
        return f"non-finite gradient norm {grad_norm}"
    if consecutive_skips >= skip_limit:
        return f"{consecutive_skips} consecutive steps where the scaler refused the gradients"
    if amp_enabled and scale < scale_floor:
        return f"loss scale collapsed to {scale}"
    if muon_skips:
        return f"Muon produced {muon_skips} non-finite updates"
    if zero_grad_streak >= zero_grad_limit:
        return f"gradient norm was exactly zero {zero_grad_streak} steps in a row"
    return ""


def train(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_stdout()
    raw = load_raw_config()
    spec = load_model_spec()

    batch_cfg = raw["batch"]
    sched_cfg = raw["schedule"]
    seq_len = args.seq_len or int(batch_cfg["seq_len"])
    if seq_len > spec.context_len:
        raise SystemExit(f"seq_len {seq_len} exceeds model context_len {spec.context_len}")
    micro_batch = args.micro_batch or int(batch_cfg["micro_batch"])
    accum = args.accum or int(batch_cfg["accum_steps"])
    eval_every = args.eval_every or int(batch_cfg["eval_every"])
    log_every = args.log_every or int(batch_cfg["log_every"])
    save_every = args.save_every or max(eval_every, 1)
    seed = args.seed if args.seed is not None else int(sched_cfg["seed"])
    mode = args.optimizer or "muon"
    grad_clip = float(sched_cfg["grad_clip"])

    device = resolve_device(args.device)
    dtype_name = args.dtype or ("fp16" if device.type == "cuda" else "fp32")
    if dtype_name == "fp16" and device.type != "cuda":
        print("fp16 autocast needs CUDA; falling back to fp32", flush=True)
        dtype_name = "fp32"
    amp_enabled = dtype_name == "fp16"
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    train_shards = TokenShards(args.train_dir, seq_len)
    val_shards = TokenShards(args.val_dir, seq_len)
    if len(train_shards) < micro_batch:
        raise SystemExit(f"only {len(train_shards)} training sequences for micro_batch {micro_batch}")
    batcher = ShardBatcher(train_shards, micro_batch, shuffle_seed=seed)
    eval_batches = args.eval_batches or int(batch_cfg["eval_batches"])
    eval_batch_size = min(micro_batch, len(val_shards)) or 1
    eval_window = eval_batches * eval_batch_size
    if len(val_shards) < 2 * eval_window:
        raise SystemExit(
            f"val holds {len(val_shards)} sequences but loss tracking and model selection each need "
            f"{eval_window}; lower eval_batches instead of scoring best.pt on the same sequences it tracks"
        )
    eval_batcher = ShardBatcher(TokenShards(args.val_dir, seq_len, offset=0), eval_batch_size, shuffle_seed=None)
    select_batcher = ShardBatcher(
        TokenShards(args.val_dir, seq_len, offset=eval_window), eval_batch_size, shuffle_seed=None
    )

    model = TransformerLM(spec, gradient_checkpoint=args.grad_ckpt).to(device=device, dtype=torch.float32)
    model.train()
    if args.init_from:
        warm = load_safetensors_model(args.init_from, device=str(device), dtype=torch.float32, spec=spec)
        model.load_state_dict(warm.state_dict())
        del warm
    muon_cfg = dict(raw["optimizer"]["muon"])
    adamw_cfg = dict(raw["optimizer"]["adamw"])
    active_cfg = muon_cfg if mode == "muon" else adamw_cfg
    if args.lr:
        active_cfg["lr"] = args.lr
    optims: list[Optimizer] = build_optimizers(model, mode=mode, muon=muon_cfg, adamw=adamw_cfg)
    compute = torch.compile(model) if args.compile else model
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    peak_reference = float(active_cfg["lr"])
    total_steps = args.steps or int(sched_cfg["total_steps"])
    warmup_steps = min(int(sched_cfg["warmup_steps"]), max(1, total_steps // 4))
    schedule = LRSchedule(
        peak_lr=peak_reference,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_fraction=float(sched_cfg["min_lr_fraction"]),
        decay=str(sched_cfg["decay"]),
    )

    start_step = 0
    if args.resume:
        path = args.resume
    elif args.dry_run:
        path = None
    else:
        path = latest_checkpoint(args.out)
    if path is not None and path.is_file():
        _, meta = load_checkpoint(path, model, optims, scaler, map_location=str(device))
        start_step = int(meta["step"])
        print(f"resumed {path} at step {start_step}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    header = {
        "event": "config",
        "git": git_state(),
        "spec": spec.to_dict(),
        "device": str(device),
        "device_name": torch.cuda.get_device_properties(device).name if device.type == "cuda" else "cpu",
        "dtype": dtype_name,
        "optimizer": mode,
        "seq_len": seq_len,
        "micro_batch": micro_batch,
        "accum_steps": accum,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "peak_lr": peak_reference,
        "seed": seed,
        "torch": torch.__version__,
        "train_sequences": len(train_shards),
        "val_sequences": len(val_shards),
        "eval_track_window": [0, eval_window],
        "eval_select_window": [eval_window, 2 * eval_window],
        "report_scope": "brainlab-ppl over the whole val dir; the in-training windows below are for tracking and selection only",
        "tokens_per_step": tokens_per_step(seq_len, micro_batch, accum),
        "compiled": bool(args.compile),
        "gradient_checkpoint": bool(args.grad_ckpt),
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, sort_keys=True) + "\n")
    print(json.dumps(header, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    data = cycle(batcher)
    step = start_step
    tokens_seen = step * header["tokens_per_step"]
    best_val = float("inf")
    window_start = time.perf_counter()
    window_tokens = 0
    ema_loss: float | None = None
    batches_in_epoch = max(len(batcher), 1)
    consecutive_skips = 0
    zero_grad_streak = 0
    muon_skips = 0
    skip_limit = int(sched_cfg["max_consecutive_skips"])
    zero_grad_limit = int(sched_cfg["max_consecutive_zero_grads"])
    scale_floor = float(sched_cfg["scale_floor"])

    while step < total_steps:
        scale_lr(optims, schedule.factor_at(step))
        accumulated = torch.zeros((), device=device)
        for _ in range(accum):
            x, y = next(data)
            ids = torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
            targets = torch.from_numpy(np.ascontiguousarray(y)).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                _, loss = compute(ids, targets)
            scaler.scale(loss / accum).backward()
            accumulated += loss.detach().float()
        window_tokens += header["tokens_per_step"]
        tokens_seen += header["tokens_per_step"]

        scale_before = scaler.get_scale()
        for optim in optims:
            scaler.unscale_(optim)
        gnorm = float(clip_grad_norm_(model.parameters(), grad_clip))
        for optim in optims:
            scaler.step(optim)
        scaler.update()
        for optim in optims:
            optim.zero_grad(set_to_none=True)

        skipped = scaler.get_scale() < scale_before
        consecutive_skips = consecutive_skips + 1 if skipped else 0
        zero_grad_streak = zero_grad_streak + 1 if gnorm == 0.0 else 0

        step += 1
        mean_loss = float(accumulated) / accum
        ema_loss = mean_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * mean_loss

        if step % log_every == 0 or step == total_steps:
            muon_skips = sum(optim.skipped_updates() for optim in optims if hasattr(optim, "skipped_updates"))
            elapsed = time.perf_counter() - window_start
            tps = window_tokens / elapsed if elapsed > 0 else 0.0
            record = {
                "event": "train",
                "step": step,
                "loss": round(mean_loss, 6),
                "loss_ema": round(float(ema_loss), 6),
                "lr": round(optims[0].param_groups[0]["lr"], 8),
                "grad_norm": round(gnorm, 4),
                "scale": scaler.get_scale(),
                "skipped_step": int(skipped),
                "consecutive_skips": consecutive_skips,
                "zero_grad_streak": zero_grad_streak,
                "muon_nonfinite_updates": muon_skips,
                "tokens_per_sec": round(tps, 1),
                "tokens_seen": tokens_seen,
                "elapsed_sec": round(time.perf_counter() - window_start, 2),
                "epoch": round(step / batches_in_epoch, 3),
                "mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3) if device.type == "cuda" else 0.0,
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            window_start = time.perf_counter()
            window_tokens = 0

        reason = divergence_reason(
            mean_loss=mean_loss,
            grad_norm=gnorm,
            skipped=skipped,
            consecutive_skips=consecutive_skips,
            zero_grad_streak=zero_grad_streak,
            scale=scaler.get_scale(),
            amp_enabled=amp_enabled,
            muon_skips=muon_skips,
            skip_limit=skip_limit,
            zero_grad_limit=zero_grad_limit,
            scale_floor=scale_floor,
        )
        if reason:
            path = save_checkpoint(args.out / f"diverged-{step:06d}.pt", model, step, optims, scaler, {"seed": seed})
            print(f"diverged at step {step}: {reason} -> {path}", flush=True)
            return 1

        if step % eval_every == 0 or step == total_steps:
            tracked = evaluate(compute, eval_batcher, device, eval_batches, amp_enabled, amp_dtype)
            tracked.update({"event": "eval", "step": step, "subset": "track", "lr": optims[0].param_groups[0]["lr"]})
            print(json.dumps(tracked, sort_keys=True, default=float), flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(tracked, sort_keys=True, default=float) + "\n")

            candidate = evaluate(compute, select_batcher, device, eval_batches, amp_enabled, amp_dtype)
            candidate.update({"event": "eval", "step": step, "subset": "select"})
            print(json.dumps(candidate, sort_keys=True, default=float), flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(candidate, sort_keys=True, default=float) + "\n")

            if candidate["val_loss"] < best_val:
                best_val = candidate["val_loss"]
                save_checkpoint(args.out / "best.pt", model, step, optims, scaler, {"seed": seed}, candidate)

        if step % save_every == 0:
            path = save_checkpoint(args.out / f"step-{step:06d}.pt", model, step, optims, scaler, {"seed": seed})
            removed = rotate(args.out, args.keep)
            if removed:
                print(f"rotated out {[str(p.name) for p in removed]}", flush=True)

    final = save_checkpoint(args.out / "last.pt", model, step, optims, scaler, {"seed": seed})
    exported = save_model(model, args.out / f"{spec.name}.safetensors", dtype=torch.float32)
    print(
        f"trained {step} steps, {tokens_seen:,} tokens, best select loss {best_val:.4f} -> {final}, {exported}",
        flush=True,
    )
    print(
        f"report the number with brainlab-ppl --model {exported} --data-dir {args.val_dir}; "
        "the two in-training windows are not a reported result",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return train(argv)


if __name__ == "__main__":
    raise SystemExit(main())
