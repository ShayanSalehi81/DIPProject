"""Generate publication-ready summary tables, plots, and visual contact sheets."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from microq_phase1.core.config import ProjectConfig
from microq_phase1.core.logging import get_logger
from microq_phase1.domain.models import AnalysisRecord, RecipeRecord
from microq_phase1.infrastructure.jsonl import read_jsonl
from microq_phase1.reporting.dataset_card import DatasetCardBuilder
from microq_phase1.reporting.statistics import (
    degradation_summary,
    restoration_delta_summary,
)

LOGGER = get_logger(__name__)


class ReportingService:
    """Build deterministic presentation artifacts from validated manifests."""

    def build(self, config: ProjectConfig) -> dict[str, str]:
        output = config.reporting.output_root
        output.mkdir(parents=True, exist_ok=True)
        degraded = self._analysis_frame(config.analysis.output_jsonl)
        summary_path = output / "degradation_summary.csv"
        self._write_frame(summary_path, degradation_summary(degraded, config))
        curve_path = output / "severity_psnr.png"
        self._severity_plot(degraded, curve_path, config.reporting.figure_dpi)
        contact_path = output / "degradation_contact_sheet.png"
        self._contact_sheet(
            config.generation.generated_manifest,
            contact_path,
            config.reporting.contact_sheet_examples_per_group,
            config.reporting.figure_dpi,
        )
        artifacts = {
            "degradation_summary": str(summary_path),
            "severity_plot": str(curve_path),
            "contact_sheet": str(contact_path),
        }
        dataset_card_path = output / "dataset_card.md"
        DatasetCardBuilder().build(config, dataset_card_path)
        artifacts["dataset_card"] = str(dataset_card_path)
        if config.analysis.restored_output_jsonl.exists():
            restored = self._analysis_frame(config.analysis.restored_output_jsonl)
            comparison_path = output / "restoration_delta_summary.csv"
            self._write_frame(
                comparison_path,
                restoration_delta_summary(degraded, restored, config),
            )
            artifacts["restoration_delta_summary"] = str(comparison_path)
        LOGGER.info("reporting_completed", artifacts=artifacts)
        return artifacts

    @staticmethod
    def _analysis_frame(path: Path) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for item in read_jsonl(path, AnalysisRecord):
            row: dict[str, object] = {
                "recipe_id": item.recipe_id,
                "source_image_id": item.source_image_id,
                "pool_role": item.pool_role,
                "modality": item.modality,
                "source_dataset": item.source_dataset,
                "degradation": item.degradation,
                "severity": item.severity,
                "step_count": item.step_count,
            }
            row.update(item.features)
            row.update(item.quality_metrics)
            row.update(item.morphology_metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _severity_plot(frame: pd.DataFrame, path: Path, dpi: int) -> None:
        order = ["weak", "medium", "strong"]
        single = frame[(frame["step_count"] == 1) & (frame["degradation"] != "clean")]
        sns.set_theme(style="whitegrid", context="talk")
        figure, axis = plt.subplots(figsize=(13, 7))
        sns.pointplot(
            data=single,
            x="severity",
            y="psnr",
            hue="degradation",
            order=order,
            errorbar=("ci", 95),
            dodge=0.35,
            ax=axis,
        )
        axis.set(title="Degradation severity calibration", xlabel="Severity", ylabel="PSNR (dB)")
        axis.legend(title="Degradation", bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.tight_layout()
        temporary = path.with_suffix(path.suffix + ".tmp")
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", format="png")
        plt.close(figure)
        os.replace(temporary, path)

    @staticmethod
    def _contact_sheet(
        manifest_path: Path,
        output_path: Path,
        examples_per_group: int,
        dpi: int,
    ) -> None:
        recipes = list(read_jsonl(manifest_path, RecipeRecord))
        selected: list[RecipeRecord] = []
        counts: dict[tuple[str, str], int] = {}
        for recipe in recipes:
            if recipe.is_clean_control or len(recipe.steps) != 1:
                continue
            key = (str(recipe.steps[0].kind), str(recipe.steps[0].severity))
            count = counts.get(key, 0)
            if count >= examples_per_group:
                continue
            counts[key] = count + 1
            selected.append(recipe)
        if not selected:
            raise ValueError("No single-degradation recipes are available for contact sheets.")
        columns = min(6, len(selected))
        rows = (len(selected) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows))
        flattened = list(getattr(axes, "flat", [axes]))
        for axis, recipe in zip(flattened, selected, strict=False):
            with Image.open(recipe.output_path) as image:
                axis.imshow(image, cmap="gray" if image.mode == "L" else None)
            step = recipe.steps[0]
            axis.set_title(f"{step.kind}\n{step.severity}", fontsize=9)
            axis.axis("off")
        for axis in flattened[len(selected) :]:
            axis.axis("off")
        figure.tight_layout()
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", format="png")
        plt.close(figure)
        os.replace(temporary, output_path)

    @staticmethod
    def _write_frame(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
