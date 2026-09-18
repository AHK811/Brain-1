"""Multimodal stack: vision, audio, video, fusion, MM wrapper."""

from brain.experimental.multimodal.vision.pipeline import VisionTower
from brain.experimental.multimodal.vision.encoder import get_vision_encoder
from brain.experimental.multimodal.vision.projector import VisionProjector
from brain.experimental.multimodal.audio import transcribe, AudioTower, process_audio
from brain.experimental.multimodal.video import process_video, VideoBundle
from brain.experimental.multimodal.mm_model import BrainMM
from brain.experimental.multimodal.unified import process_media, MediaBundle
from brain.experimental.multimodal.fusion.interleave import (
    build_interleaved_prompt, inject_vision_into_embeds, IMAGE_PAD, IMAGE_START, IMAGE_END,
)

__all__ = [
    "VisionTower", "get_vision_encoder", "VisionProjector",
    "transcribe", "AudioTower", "process_audio",
    "process_video", "VideoBundle",
    "BrainMM", "process_media", "MediaBundle",
    "build_interleaved_prompt", "inject_vision_into_embeds",
    "IMAGE_PAD", "IMAGE_START", "IMAGE_END",
]
