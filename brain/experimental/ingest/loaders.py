"""Per-format loaders → ContentBlock lists."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Callable, Optional

from brain.experimental.ingest.types import ContentBlock, IngestResult, Modality
from brain.experimental.ocr.engines import OCREngine, get_ocr_engine


Loader = Callable[[Path, dict], IngestResult]


def _meta(path: Path, **extra) -> ContentBlock:
    return ContentBlock(
        modality=Modality.META,
        text=f"file={path.name} size={path.stat().st_size if path.exists() else 0}",
        source=str(path),
        mime=extra.get("mime", ""),
        meta=extra,
    )


def load_text(path: Path, opts: dict | None = None) -> IngestResult:
    opts = opts or {}
    try:
        text = path.read_text(encoding=opts.get("encoding", "utf-8"), errors="ignore")
    except Exception as e:
        return IngestResult(source=str(path), errors=[str(e)])
    return IngestResult(
        source=str(path),
        blocks=[_meta(path, mime="text/plain"), ContentBlock(Modality.TEXT, text=text, source=str(path))],
    )


def load_pdf(path: Path, opts: dict | None = None) -> IngestResult:
    opts = opts or {}
    blocks: list[ContentBlock] = [_meta(path, mime="application/pdf")]
    errors: list[str] = []
    # Try pypdf then pdfplumber
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            t = (page.extract_text() or "").strip()
            if t:
                blocks.append(ContentBlock(Modality.TEXT, text=t, source=str(path), page=i + 1))
        if len(blocks) == 1:
            errors.append("pypdf extracted no text (may be scanned — try OCR)")
        return IngestResult(source=str(path), blocks=blocks, errors=errors)
    except Exception as e1:
        errors.append(f"pypdf: {e1}")
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = (page.extract_text() or "").strip()
                if t:
                    blocks.append(ContentBlock(Modality.TEXT, text=t, source=str(path), page=i + 1))
        return IngestResult(source=str(path), blocks=blocks, errors=errors)
    except Exception as e2:
        errors.append(f"pdfplumber: {e2}")
    return IngestResult(source=str(path), blocks=blocks, errors=errors)


def load_docx(path: Path, opts: dict | None = None) -> IngestResult:
    blocks = [_meta(path, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")]
    try:
        import docx
        doc = docx.Document(str(path))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for ti, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                rows.append(" | ".join(c.text.strip() for c in row.cells))
            if rows:
                blocks.append(ContentBlock(Modality.TABLE, text="\n".join(rows), source=str(path), meta={"table": ti}))
        if paras:
            blocks.append(ContentBlock(Modality.TEXT, text="\n\n".join(paras), source=str(path)))
        return IngestResult(source=str(path), blocks=blocks)
    except Exception as e:
        return IngestResult(source=str(path), blocks=blocks, errors=[str(e)])


def load_pptx(path: Path, opts: dict | None = None) -> IngestResult:
    blocks = [_meta(path, mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")]
    try:
        from pptx import Presentation
        prs = Presentation(str(path))
        for i, slide in enumerate(prs.slides):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            if texts:
                blocks.append(ContentBlock(
                    Modality.TEXT, text="\n".join(texts), source=str(path), page=i + 1, meta={"slide": i + 1}
                ))
        return IngestResult(source=str(path), blocks=blocks)
    except Exception as e:
        return IngestResult(source=str(path), blocks=blocks, errors=[str(e)])


def load_xlsx(path: Path, opts: dict | None = None) -> IngestResult:
    opts = opts or {}
    blocks = [_meta(path, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")]
    max_rows = int(opts.get("max_rows", 200))
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        for sheet in wb.worksheets:
            rows = []
            for ri, row in enumerate(sheet.iter_rows(values_only=True)):
                if ri >= max_rows:
                    rows.append("... [truncated]")
                    break
                cells = ["" if c is None else str(c) for c in row]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(ContentBlock(
                    Modality.TABLE, text="\n".join(rows), source=str(path), meta={"sheet": sheet.title}
                ))
        return IngestResult(source=str(path), blocks=blocks)
    except Exception as e:
        return IngestResult(source=str(path), blocks=blocks, errors=[str(e)])


def load_csv(path: Path, opts: dict | None = None) -> IngestResult:
    opts = opts or {}
    blocks = [_meta(path, mime="text/csv")]
    max_rows = int(opts.get("max_rows", 200))
    try:
        with path.open(encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    rows.append("... [truncated]")
                    break
                rows.append(" | ".join(row))
        blocks.append(ContentBlock(Modality.TABLE, text="\n".join(rows), source=str(path)))
        return IngestResult(source=str(path), blocks=blocks)
    except Exception as e:
        return IngestResult(source=str(path), blocks=blocks, errors=[str(e)])


def load_image(path: Path, opts: dict | None = None) -> IngestResult:
    opts = opts or {}
    ocr: OCREngine = opts.get("ocr") or get_ocr_engine(opts.get("ocr_prefer"))
    blocks = [_meta(path, mime="image/*"), ContentBlock(
        Modality.IMAGE, text="", source=str(path), image_path=str(path), meta={"ocr_engine": ocr.name}
    )]
    try:
        text = ocr.extract(path)
        blocks[-1].text = text or "[no text detected]"
    except Exception as e:
        return IngestResult(source=str(path), blocks=blocks, errors=[str(e)])
    return IngestResult(source=str(path), blocks=blocks)


def load_audio(path: Path, opts: dict | None = None) -> IngestResult:
    """ASR via optional openai-whisper or faster-whisper; else stub."""
    opts = opts or {}
    blocks = [_meta(path, mime="audio/*"), ContentBlock(
        Modality.AUDIO, text="", source=str(path), audio_path=str(path)
    )]
    try:
        import whisper
        model_name = opts.get("whisper_model", "base")
        model = whisper.load_model(model_name)
        result = model.transcribe(str(path))
        blocks[-1].text = (result.get("text") or "").strip()
        blocks[-1].meta["asr"] = "whisper"
        return IngestResult(source=str(path), blocks=blocks)
    except Exception as e1:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(opts.get("whisper_model", "base"))
            segments, _ = model.transcribe(str(path))
            blocks[-1].text = " ".join(s.text for s in segments).strip()
            blocks[-1].meta["asr"] = "faster-whisper"
            return IngestResult(source=str(path), blocks=blocks)
        except Exception as e2:
            blocks[-1].text = "[asr unavailable — install openai-whisper or faster-whisper]"
            return IngestResult(source=str(path), blocks=blocks, errors=[str(e1), str(e2)])


def load_video(path: Path, opts: dict | None = None) -> IngestResult:
    """Sample keyframes + optional audio ASR."""
    opts = opts or {}
    blocks = [_meta(path, mime="video/*")]
    errors: list[str] = []
    n_frames = int(opts.get("n_frames", 4))
    # Frame sampling with opencv if available
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = [int(i * total / max(n_frames, 1)) for i in range(n_frames)] if total else list(range(n_frames))
        out_dir = Path(opts.get("frame_dir", "/tmp/brain_v04_frames"))
        out_dir.mkdir(parents=True, exist_ok=True)
        ocr = opts.get("ocr") or get_ocr_engine(opts.get("ocr_prefer"))
        for fi, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            fp = out_dir / f"{path.stem}_f{fi}.jpg"
            cv2.imwrite(str(fp), frame)
            try:
                txt = ocr.extract(fp)
            except Exception:
                txt = ""
            blocks.append(ContentBlock(
                Modality.IMAGE, text=txt or f"[frame {fi}]", source=str(path),
                page=fi, image_path=str(fp), meta={"frame_index": idx}
            ))
        cap.release()
    except Exception as e:
        errors.append(f"frames: {e}")
        blocks.append(ContentBlock(Modality.VIDEO, text="[video frame extract unavailable — install opencv-python]",
                                   source=str(path), video_path=str(path)))
    # Optional: extract audio track ASR would go here (ffmpeg) — left as host enhancement
    return IngestResult(source=str(path), blocks=blocks, errors=errors)

