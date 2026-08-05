"""Pinned export adapter for the official evaluation-only Micro-Bench release."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from hashlib import sha256
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
        grouped: dict[str, dict[str, Any]] = {}
        excluded_licenses: Counter[str] = Counter()
        for index, item in enumerate(dataset):
            if index >= limit:
                break
            license_name = str(item.get("license") or "unknown")
            if (
                config.included_licenses is not None
                and license_name not in config.included_licenses
            ):
                excluded_licenses[license_name] += 1
                continue
            image_id = str(item["image_id"])
            image = item["image"]
            digest = self._image_digest(image)
            properties = self._image_properties(item, image)
            annotation = self._benchmark_annotation(item, index)
            source_dataset = str(item.get("dataset") or "unknown")

            if image_id in grouped:
                group = grouped[image_id]
                if group["digest"] != digest:
                    raise ValueError(
                        f"Micro-Bench image_id {image_id!r} refers to different image pixels."
                    )
                if group["properties"] != properties:
                    differing = sorted(
                        key
                        for key, value in properties.items()
                        if group["properties"][key] != value
                    )
                    raise ValueError(
                        f"Micro-Bench image_id {image_id!r} has inconsistent image "
                        f"metadata: {', '.join(differing)}"
                    )
                if source_dataset not in group["source_datasets"]:
                    group["source_datasets"].append(source_dataset)
                group["annotations"].append(annotation)
                group["polygons"].append(item.get("polygon"))
            else:
                image_path = config.output_root / "images" / f"{image_id}.png"
                self._save_lossless(image, image_path, expected_digest=digest)
                grouped[image_id] = {
                    "image_path": image_path,
                    "digest": digest,
                    "properties": properties,
                    "source_datasets": [source_dataset],
                    "annotations": [annotation],
                    "polygons": [item.get("polygon")],
                    "width": image.width,
                    "height": image.height,
                }
            if (index + 1) % 500 == 0:
                LOGGER.info(
                    "microbench_export_progress",
                    completed=index + 1,
                    total=min(limit, len(dataset)),
                )

        if config.included_licenses is not None:
            LOGGER.info(
                "microbench_license_filter_applied",
                included_licenses=sorted(config.included_licenses),
                excluded_by_license=dict(sorted(excluded_licenses.items())),
                included_source_rows=sum(len(group["annotations"]) for group in grouped.values()),
            )

        for image_id, group in grouped.items():
            properties = group["properties"]
            mask_path = self._masks.export(
                group["polygons"],
                width=group["width"],
                height=group["height"],
                output_path=config.output_root / "masks" / f"{image_id}.png",
            )
            yield ImageRecord(
                image_id=image_id,
                image_path=group["image_path"].resolve(),
                source_dataset=group["source_datasets"][0],
                license=properties["license"],
                pool_role=PoolRole.EVALUATION,
                modality=properties["modality"],
                submodality=properties["submodality"],
                domain=properties["domain"],
                subdomain=properties["subdomain"],
                stain=properties["stain"],
                microns_per_pixel=properties["microns_per_pixel"],
                mask_path=mask_path,
                group_id=image_id,
                is_multichannel=properties["is_multichannel"],
                bit_depth=properties["bit_depth"],
                metadata={
                    "source_datasets": group["source_datasets"],
                    "benchmark_annotations": group["annotations"],
                    "dataset_revision": config.revision,
                },
            )

    def _image_properties(self, item: dict[str, Any], image: Image.Image) -> dict[str, Any]:
        """Return fields that must agree when task rows reuse one source image."""
        return {
            "license": str(item.get("license") or "unknown"),
            "modality": self._map_modality(item.get("modality")),
            "submodality": self._optional_text(item.get("submodality")),
            "domain": self._optional_text(item.get("domain")),
            "subdomain": self._optional_text(item.get("subdomain")),
            "stain": self._optional_text(item.get("stain")),
            "microns_per_pixel": self._positive_float(item.get("microns_per_pixel")),
            "is_multichannel": self._is_multichannel(image),
            "bit_depth": self._bit_depth(image),
        }

    @staticmethod
    def _benchmark_annotation(item: dict[str, Any], index: int) -> dict[str, Any]:
        """Preserve every task row associated with a physical benchmark image."""
        return {
            "source_row_index": index,
            "source_dataset": str(item.get("dataset") or "unknown"),
            "label": item.get("label"),
            "label_name": item.get("label_name"),
            "questions": item.get("questions"),
            "captions": item.get("captions"),
            "synthetic": item.get("synthetic"),
            "bbox": item.get("bbox"),
        }

    @staticmethod
    def _image_digest(image: Image.Image) -> str:
        digest = sha256()
        digest.update(image.mode.encode("utf-8"))
        digest.update(str(image.size).encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()

    @classmethod
    def _save_lossless(
        cls,
        image: Image.Image,
        path: Path,
        *,
        expected_digest: str,
    ) -> None:
        if path.is_file():
            try:
                with Image.open(path) as existing:
                    existing.load()
                    if cls._image_digest(existing) == expected_digest:
                        return
            except (OSError, ValueError):
                # A stale or damaged file is replaced atomically below.
                pass
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
        if not isinstance(value, str | bytes | int | float):
            return None
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
