import torch
from brain.experimental.multimodal.audio import AudioTower, MelStubEncoder, transcribe
from brain.experimental.multimodal.audio.asr import ASRResult


def test_audio_tower_shape():
    tower = AudioTower(hidden_size=384, audio_dim=512, n_tokens=16)
    # (B, 1, T) fake waveform/mel
    x = torch.randn(2, 1, 16000)
    y = tower(x)
    assert y.shape == (2, 16, 384)


def test_mel_stub_encoder():
    enc = MelStubEncoder(dim=128, n_tokens=8)
    y = enc(torch.randn(1, 1, 4000))
    assert y.shape == (1, 8, 128)


def test_transcribe_missing_file():
    r = transcribe("/tmp/definitely_missing_brain_audio.wav")
    assert isinstance(r, ASRResult)
    assert r.backend in ("none", "stub", "error", "whisper", "faster-whisper")


def test_video_import():
    from brain.experimental.multimodal.video import process_video, VideoBundle
    b = VideoBundle(path="x.mp4")
    assert "video" in b.to_prompt().lower()
