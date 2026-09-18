#!/usr/bin/env python3
"""Run a quick verification suite for Brain v0.4."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> int:
    fails = 0
    def check(name, fn):
        nonlocal fails
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            fails += 1

    def core():
        from brain.core.config import BrainConfig
        from brain.model.brain_model import BrainForCausalLM
        c = BrainConfig.from_yaml(str(ROOT / "configs/model/brain_0_4_mm.yaml"))
        assert BrainForCausalLM(c).count_parameters()["total"] == 33_236_352

    def phase_a():
        from brain.experimental.ingest import load_any
        import tempfile
        p = Path(tempfile.mkdtemp()) / "t.txt"
        p.write_text("ok")
        assert load_any(p).blocks

    def phase_b():
        import torch
        from brain.experimental.multimodal import VisionTower
        from brain.experimental.multimodal.vision.encoder import StubVisionEncoder
        t = VisionTower(hidden_size=64, encoder=StubVisionEncoder(768, 8))
        assert t(torch.randn(1,3,64,64)).shape[-1] == 64

    def phase_c():
        import torch
        from brain.experimental.multimodal import AudioTower
        assert AudioTower(64, 128, 8)(torch.randn(1,1,500)).shape == (1,8,64)

    def phase_d():
        from brain.experimental.multimodal import VideoBundle
        assert "video" in VideoBundle("x.mp4").to_prompt()

    def phase_e():
        from brain.experimental.mm_sft import MMSFTExample, write_sample_sft_jsonl, load_mm_sft_jsonl
        import tempfile
        p = Path(tempfile.mkdtemp()) / "s.jsonl"
        write_sample_sft_jsonl(p)
        rows = load_mm_sft_jsonl(p)
        assert len(rows) >= 2
        assert "2+2" in rows[1].build_prompt() or rows[1].answer == "4"

    for n,f in [("core",core),("A_ingest",phase_a),("B_vision",phase_b),
                ("C_audio",phase_c),("D_video",phase_d),("E_mm_sft",phase_e)]:
        check(n,f)
    print("result:", "OK" if fails==0 else f"{fails} failed")
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())
