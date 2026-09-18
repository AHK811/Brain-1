import torch
from brain.experimental.multimodal import VisionTower, VisionProjector, BrainMM, build_interleaved_prompt
from brain.experimental.multimodal.vision.encoder import StubVisionEncoder


def test_projector_shape():
    p = VisionProjector(vision_dim=768, hidden_size=384)
    x = torch.randn(2, 16, 768)
    y = p(x)
    assert y.shape == (2, 16, 384)
    assert p.num_parameters() > 0


def test_stub_tower():
    tower = VisionTower(hidden_size=384, encoder=StubVisionEncoder(dim=768, n_tokens=32))
    imgs = torch.randn(1, 3, 224, 224)
    out = tower(imgs)
    assert out.shape[0] == 1 and out.shape[-1] == 384


def test_interleaved_prompt():
    s = build_interleaved_prompt("describe this", n_image_tokens=4)
    assert "<|image_start|>" in s and "<|image_pad|>" in s


def test_brain_mm_with_tiny_lm():
    from brain.core.config import BrainConfig
    from brain.model.brain_model import BrainForCausalLM
    cfg = BrainConfig(
        vocab_size=256, hidden_size=64, num_layers=1, num_heads=4,
        num_kv_heads=2, intermediate_size=128, max_position_embeddings=64,
    )
    lm = BrainForCausalLM(cfg)
    mm = BrainMM(lm, vision=VisionTower(hidden_size=64, vision_dim=768, n_tokens=8,
                                        encoder=StubVisionEncoder(dim=768, n_tokens=8)), hidden_size=64)
    ids = torch.randint(0, 256, (1, 16))
    imgs = torch.randn(1, 3, 64, 64)
    out = mm(ids, images=imgs)
    assert "logits" in out or hasattr(out, "logits")
