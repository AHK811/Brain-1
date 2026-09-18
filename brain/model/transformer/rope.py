"""
Rotary Position Embeddings with NTK / linear / YaRN / PI scaling.

NTK, linear, and PI (position interpolation) below are the real published
formulations. "yarn" is NOT the real YaRN algorithm (Peng et al.) -- true
YaRN interpolates different frequency bands differently (NTK-by-parts) and
derives its attention-temperature correction from that band-wise ramp. This
is a much simpler stand-in (uniform theta multiply + a single global
position ramp + one scalar magnitude correction) that behaves reasonably
but should not be assumed to reproduce YaRN's actual accuracy/extrapolation
properties. Treat scaling_type="yarn" as experimental until it's replaced
with a real per-band implementation and validated against it.
"""
from __future__ import annotations
from typing import Optional
import math
import torch


class RotaryEmbedding:
    """Precomputes cos/sin tables; supports NTK-aware, linear, YaRN, PI scaling."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        theta: float = 500000.0,
        scaling_type: Optional[str] = None,
        scaling_factor: float = 1.0,
    ):
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = float(theta)
        self.scaling_type = (scaling_type or "").lower() or None
        self.scaling_factor = max(float(scaling_factor), 1.0)
        self.effective_theta = self._compute_effective_theta()
        self.inv_freq = self._make_inv_freq()
        self._build_tables(max_position_embeddings)

    def _compute_effective_theta(self) -> float:
        if self.scaling_type == "ntk" and self.scaling_factor > 1.0:
            hd = self.head_dim
            return self.theta * (self.scaling_factor ** (hd / (hd - 2)))
        if self.scaling_type == "yarn" and self.scaling_factor > 1.0:
            # YaRN-style base adjustment (simplified)
            return self.theta * self.scaling_factor
        return self.theta

    def _make_inv_freq(self) -> torch.Tensor:
        hd = self.head_dim
        if self.scaling_type in ("ntk", "yarn") and self.scaling_factor > 1.0:
            theta = self.effective_theta
        else:
            theta = self.theta
        return 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))

    def _build_tables(self, max_pos: int) -> None:
        positions = torch.arange(max_pos).float()
        st = self.scaling_type
        if st in ("linear", "pi") and self.scaling_factor > 1.0:
            # Positional interpolation: compress positions into original range
            positions = positions / self.scaling_factor
        elif st == "yarn" and self.scaling_factor > 1.0:
            # Mild position mix (simplified YaRN ramp)
            ramp = torch.clamp((positions - max_pos / self.scaling_factor) / (max_pos + 1e-6), 0.0, 1.0)
            positions = positions * (1.0 - 0.1 * ramp)
        freqs = torch.outer(positions, self.inv_freq)
        # YaRN attention temperature correction factor (optional scale on cos/sin magnitude)
        if st == "yarn" and self.scaling_factor > 1.0:
            mscale = 0.1 * math.log(self.scaling_factor) + 1.0
            self.cos = (freqs.cos() * mscale)
            self.sin = (freqs.sin() * mscale)
        else:
            self.cos = freqs.cos()
            self.sin = freqs.sin()
        self.max_position_embeddings = max_pos

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "RotaryEmbedding":
        self.inv_freq = self.inv_freq.to(device=device)
        self.cos = self.cos.to(device=device, dtype=dtype) if dtype else self.cos.to(device)
        self.sin = self.sin.to(device=device, dtype=dtype) if dtype else self.sin.to(device)
        return self

    def get(self, seq_len: int, position_offset: int = 0):
        end = position_offset + seq_len
        if end > self.max_position_embeddings:
            # Graceful extension -- must preserve device/dtype, since
            # _build_tables() recreates self.cos/self.sin from scratch via
            # torch.arange(...), which defaults to CPU float32 regardless of
            # where the model actually lives. Without this, extending
            # context on a GPU-resident model would silently rebuild the
            # RoPE tables on CPU and crash (or worse, mismatch) on the next
            # attention call.
            device, dtype = self.cos.device, self.cos.dtype
            self._build_tables(end + 64)
            self.inv_freq = self.inv_freq.to(device=device)
            self.cos = self.cos.to(device=device, dtype=dtype)
            self.sin = self.sin.to(device=device, dtype=dtype)
        cos = self.cos[position_offset:end]
        sin = self.sin[position_offset:end]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (batch, heads, seq, head_dim); cos/sin: (seq, head_dim/2)."""
    # pair dims
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,seq,hd/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    # rotate
    rot0 = x1 * cos - x2 * sin
    rot1 = x1 * sin + x2 * cos
    out = torch.stack((rot0, rot1), dim=-1).flatten(-2)
    return out.to(dtype=x.dtype)
