"""Pinned export adapter for the official evaluation-only Micro-Bench release."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image

from microq_phase1.core.config import MicroBenchExportConfig
from microq_phase1.core.logging import get_logger
from microq_phase1.domain.models import ImageRecord, Modality, PoolRole
from microq_phase1.infrastructure.jsonl import write_jsonl_atomic
from microq_phase1.infrastructure.polygons import PolygonMaskExporter

LOGGER = get_logger(__name__)


class MicroBenchExporter:
    """Export Hugging Face Arrow images into an immutable local evaluation catalog."""

    def __init__(self) -> None:
        self._masks = PolygonMaskExporter()

    def export(self, config: MicroBenchExportConfig) -> int:
        dataset = load_dataset(
            config.dataset_id,
            split=config.split,
            revision=config.revision,
            cache_dir=str(config.cache_dir),
        )
        config.output_root.mkdir(parents=True, exist_ok=True)
        records = self._iter_records(dataset, config)
        materialized = list(records)
        write_jsonl_atomic(config.catalog_path, materialized)
        LOGGER.info(
            "microbench_exported",
            dataset_id=config.dataset_id,
            revision=config.revision,
            records=len(materialized),
            catalog=str(config.catalog_path),
        )
        return len(materialized)

    def _iter_records(
        self,
        dataset: Any,
        config: MicroBenchExportConfig,
    ) -> Iterator[ImageRecord]:
        limit = config.max_records or len(dataset)
        for index, item in enumerate(dataset):
            if index >= limit:
                return
            image_id = str(item["image_id"])
            image_path = config.output_root / "images" / f"{image_id}.png"
            image = item["image"]
            self._save_lossless(image, image_path)
            mask_path = self._masks.export(
                item.get("polygon"),
                width=image.width,
                height=image.height,
                output_path=config.output_root / "masks" / f"{image_id}.png",
            )
            if (index + 1) % 500 == 0:
                LOGGER.info(
                    "microbench_export_progress",
                    completed=index + 1,
                    total=min(limit, len(dataset)),
                )
            yield ImageRecord(
                image_id=image_id,
                image_path=image_path.resolve(),
                source_dataset=str(item.get("dataset") or "unknown"),
                license=str(item.get("license") or "unknown"),
                pool_role=PoolRole.EVALUATION,
                modality=self._map_modality(item.get("modality")),
                submodality=self._optional_text(item.get("submodality")),
                domain=self._optional_text(item.get("domain")),
                subdomain=self._optional_text(item.get("subdomain")),
                stain=self._optional_text(item.get("stain")),
                microns_per_pixel=self._positive_float(item.get("microns_per_pixel")),
                mask_path=mask_path,
                group_id=str(item.get("image_id")),
                is_multichannel=self._is_multichannel(item["image"]),
                bit_depth=self._bit_depth(item["image"]),
                metadata={
                    "label": item.get("label"),
                    "label_name": item.get("label_name"),
                    "questions": item.get("questions"),
                    "captions": item.get("captions"),
                    "synthetic": item.get("synthetic"),
                    "bbox": item.get("bbox"),
                    "dataset_revision": config.revision,
                },
            )

    @staticmethod
    def _save_lossless(image: Image.Image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        image.save(temporary, format="PNG")
        os.replace(temporary, path)

    @staticmethod
    def _map_modality(value: object) -> Modality:
        normalized = str(value or "").lower()
        if "fluorescence" in normalized:
            return Modality.FLUORESCENCE
        if "electron" in normalized:
            return Modality.ELECTRON
        if "light" in normalized or "brightfield" in normalized or "darkfield" in normalized:
            return Modality.LIGHT
        return Modality.UNKNOWN

    @staticmethod
    def _optional_text(value: object) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None

    @staticmethod
    def _positive_float(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _is_multichannel(image: Image.Image) -> bool:
        return image.mode in {"RGB", "RGBA"} and len(image.getbands()) >= 3

    @staticmethod
    def _bit_depth(image: Image.Image) -> int:
        return 16 if image.mode.startswith("I;16") else 8
