"""End-to-end orchestration after source catalogs have been prepared."""

from __future__ import annotations

from typing import Any

from microq_phase1.core.config import ProjectConfig


class PhaseOneWorkflow:
    """Run the complete local pipeline in its required scientific order."""

    def run(self, config: ProjectConfig, services: dict[str, Any]) -> None:
        services["audit"].audit(config)
        services["planner"].plan(config)
        services["generation"].materialize(config)
        services["analysis"].analyze(config)
        if config.restoration.enabled:
            services["restoration"].restore(config)
            services["analysis"].analyze(
                config,
                manifest_path=config.restoration.manifest_path,
                output_jsonl=config.analysis.restored_output_jsonl,
                output_csv=config.analysis.restored_output_csv,
            )
        services["validation"].validate(config)
        services["baseline"].run(
            analysis_path=config.analysis.output_jsonl,
            output_dir=config.analysis.baseline_output_dir,
        )
        services["reporting"].build(config)
