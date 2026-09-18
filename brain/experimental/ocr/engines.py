"""
Open-source OCR backends (optional deps).

Priority:
  1. pytesseract + Pillow  (classic, widely available)
  2. easyocr               (stronger, heavier)
  3. paddleocr             (optional)
  4. stub                  (always works)

Install examples:
  pip install pytesseract pillow
  sudo apt-get install tesseract-ocr   # system binary
  pip install easyocr
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol


class OCREngine(Protocol):
    name: str
    def extract(self, image_path: str | Path) -> str: ...


class StubOCR:
    name = "stub"
    def extract(self, image_path: str | Path) -> str:
        return f"[ocr-stub] no engine installed for {image_path}"


class TesseractOCR:
    name = "tesseract"
    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def extract(self, image_path: str | Path) -> str:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        return (pytesseract.image_to_string(img, lang=self.lang) or "").strip()


class EasyOCREngine:
    name = "easyocr"
    def __init__(self, langs: Optional[list[str]] = None):
        import easyocr
        self._reader = easyocr.Reader(langs or ["en"], gpu=False)

    def extract(self, image_path: str | Path) -> str:
        rows = self._reader.readtext(str(image_path), detail=0, paragraph=True)
        if isinstance(rows, list):
            return "\n".join(str(r) for r in rows).strip()
        return str(rows).strip()


def get_ocr_engine(prefer: Optional[str] = None) -> OCREngine:
    """Pick the best available OCR engine."""
    order = []
    if prefer:
        order.append(prefer)
    order.extend(["tesseract", "easyocr", "paddle", "stub"])
    seen = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        try:
            if name == "tesseract":
                eng = TesseractOCR()
                # probe
                import pytesseract  # noqa: F401
                return eng
            if name == "easyocr":
                return EasyOCREngine()
            if name == "paddle":
                # lazy optional
                from paddleocr import PaddleOCR  # type: ignore
                class _Paddle:
                    name = "paddle"
                    def __init__(self):
                        self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                    def extract(self, image_path):
                        out = self._ocr.ocr(str(image_path), cls=True)
                        lines = []
                        for page in out or []:
                            for line in page or []:
                                if line and len(line) >= 2:
                                    lines.append(str(line[1][0]))
                        return "\n".join(lines)
                return _Paddle()
        except Exception:
            continue
    return StubOCR()
