"""Catalog repositories and relative-path resolution."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from microq_phase1.core.errors import CatalogError
from microq_phase1.domain.models import ImageRecord
from microq_phase1.infrastructure.jsonl import read_jsonl


class JsonlImageCatalog:
    """Read validated image records from a portable JSONL catalog."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self, *, require_files: bool = True) -> Iterator[ImageRecord]:
        if not self.path.exists():
            raise CatalogError(f"Catalog does not exist: {self.path}")
        for record in read_jsonl(self.path, ImageRecord):
            resolved = self._resolve_record_paths(record)
            if require_files:
                self._require_files(resolved)
            yield resolved

    def _resolve_record_paths(self, record: ImageRecord) -> ImageRecord:
        base = self.path.resolve().parent
        image_path = record.image_path
        mask_path = record.mask_path
        if not image_path.is_absolute():
            image_path = (base / image_path).resolve()
        if mask_path is not None and not mask_path.is_absolute():
            mask_path = (base / mask_path).resolve()
        return record.model_copy(update={"image_path": image_path, "mask_path": mask_path})

    @staticmethod
    def _require_files(record: ImageRecord) -> None:
        if not record.image_path.is_file():
            raise CatalogError(f"Image is missing for {record.image_id}: {record.image_path}")
        if record.mask_path is not None and not record.mask_path.is_file():
            raise CatalogError(f"Mask is missing for {record.image_id}: {record.mask_path}")

