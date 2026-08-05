"""Command-line interface for every explicit Phase 1 stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from microq_phase1.application.splitting import ExternalCatalogSplitter
from microq_phase1.application.workflow import PhaseOneWorkflow
from microq_phase1.cli_support import execute
from microq_phase1.core.config import ProjectConfig
from microq_phase1.infrastructure.microbench import MicroBenchExporter

app = typer.Typer(
    name="microq-phase1", help="Build the Phase 1 degradation benchmark.", no_args_is_help=True
)


@app.command("check-config")
def check_config(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Validate configuration and strict pool roles without opening dataset catalogs."""

    def action(config: ProjectConfig, services: dict[str, Any]) -> None:
        services["config"].validate(config)

    execute(config_path, "check-config", action)


@app.command("export-microbench")
def export_microbench(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Download/export the pinned official benchmark into an evaluation-only catalog."""
    execute(
        config_path,
        "export-microbench",
        lambda config, services: MicroBenchExporter().export(config.microbench_export),
    )


@app.command("split-external-catalog")
def split_external_catalog(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    input_catalog: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    training_output: Path = typer.Option(..., "--training-output", dir_okay=False),
    development_output: Path = typer.Option(..., "--development-output", dir_okay=False),
    development_fraction: float = typer.Option(0.15, "--development-fraction"),
    group_by_source_dataset: bool = typer.Option(
        True,
        "--group-by-source-dataset/--group-by-image-group",
    ),
) -> None:
    """Split a curated external catalog while keeping provenance groups intact."""

    def action(config: ProjectConfig, services: dict[str, Any]) -> tuple[int, int]:
        del services
        return ExternalCatalogSplitter().split(
            input_catalog=input_catalog,
            training_output=training_output,
            development_output=development_output,
            development_fraction=development_fraction,
            seed=config.generation.master_seed,
            group_by_source_dataset=group_by_source_dataset,
        )

    execute(config_path, "split-external-catalog", action)


@app.command("audit-pools")
def audit_pools(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Reject role mistakes and exact/perceptual train-evaluation overlap."""
    execute(
        config_path,
        "audit-pools",
        lambda config, services: services["audit"].audit(config),
    )


@app.command("plan-recipes")
def plan_recipes(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Create the deterministic balanced recipe manifest."""
    execute(
        config_path,
        "plan-recipes",
        lambda config, services: services["planner"].plan(config),
    )


@app.command("materialize")
def materialize(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Materialize planned degradation recipes."""
    execute(
        config_path,
        "materialize",
        lambda config, services: services["generation"].materialize(config),
    )


@app.command("analyze")
def analyze(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Compute degradation fingerprints and full-reference metrics."""
    execute(
        config_path,
        "analyze",
        lambda config, services: services["analysis"].analyze(config),
    )


@app.command("restore")
def restore(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Apply matched classical restorations and write a restored manifest."""
    execute(
        config_path,
        "restore",
        lambda config, services: services["restoration"].restore(config),
    )


@app.command("analyze-restored")
def analyze_restored(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Measure restored artifacts using the same DIP protocol."""

    def action(config: ProjectConfig, services: dict[str, Any]) -> int:
        return int(
            services["analysis"].analyze(
                config,
                manifest_path=config.restoration.manifest_path,
                output_jsonl=config.analysis.restored_output_jsonl,
                output_csv=config.analysis.restored_output_csv,
            )
        )

    execute(config_path, "analyze-restored", action)


@app.command("validate")
def validate(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Run dataset completeness, balance, and severity quality gates."""
    execute(
        config_path,
        "validate",
        lambda config, services: services["validation"].validate(config),
    )


@app.command("run-baseline")
def run_baseline(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Fit the DIP fingerprint readout externally and test on Micro-Bench."""
    execute(
        config_path,
        "run-baseline",
        lambda config, services: services["baseline"].run(
            analysis_path=config.analysis.output_jsonl,
            output_dir=config.analysis.baseline_output_dir,
        ),
    )


@app.command("build-report")
def build_report(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Create summary tables, severity curves, restoration deltas, and contact sheets."""
    execute(
        config_path,
        "build-report",
        lambda config, services: services["reporting"].build(config),
    )


@app.command("run-all")
def run_all(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Run all local Phase 1 stages after both catalogs have been prepared."""
    execute(
        config_path,
        "run-all",
        lambda config, services: PhaseOneWorkflow().run(config, services),
    )
