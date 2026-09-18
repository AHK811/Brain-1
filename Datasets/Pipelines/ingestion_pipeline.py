from __future__ import annotations
from pathlib import Path
from Datasets.Processing.cleaning import clean_text
from Datasets.Processing.chunking import chunk_text
from Datasets.Registry.dataset_registry import DatasetRegistry, DatasetMeta
class IngestionPipeline:
    def __init__(self, registry: DatasetRegistry | None = None):
        self.registry = registry or DatasetRegistry()
    def ingest_text_file(self, path: str | Path, name: str | None = None, chunk_size: int = 0) -> DatasetMeta:
        path = Path(path)
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        records = chunk_text(text, size=chunk_size) if chunk_size > 0 else [text]
        meta = DatasetMeta(name=name or path.stem, path=str(path), kind="text", n_records=len(records))
        self.registry.register(meta)
        meta.meta["records_preview"] = records[:3]
        return meta
