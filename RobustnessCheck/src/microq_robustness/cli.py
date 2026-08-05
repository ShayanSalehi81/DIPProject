"""Command-line entry points for preparation, evaluation, and analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from microq_robustness.config import Settings
from microq_robustness.errors import RobustnessError
from microq_robustness.evaluator import RobustnessEvaluator
from microq_robustness.logging_utils import configure_logging
from microq_robustness.sampling import PairedSampleSelector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def _settings() -> Settings:
    settings = Settings.load(PROJECT_ROOT)
    configure_logging(settings.project_root)
    return settings


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a frozen paired robustness cohort.")
    parser.add_argument(
        "--force", action="store_true", help="Replace the current cohort intentionally."
    )
    args = parser.parse_args()
    try:
        samples = PairedSampleSelector(_settings()).prepare(force=args.force)
        print(f"Prepared {len(samples)} paired samples ({len(samples) * 3} requests).")
    except RobustnessError as exc:
        LOGGER.error("Preparation failed: %s", exc)
        raise SystemExit(2) from exc


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Run clean/degraded/restored VLM evaluation.")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Archive existing results and evaluate the frozen cohort from the start.",
    )
    args = parser.parse_args()
    try:
        metadata = RobustnessEvaluator(_settings()).run(restart=args.restart)
        print(
            f"Evaluation complete: {metadata['new_requests']} new requests; "
            f"results are in {_settings().results_path}."
        )
    except RobustnessError as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        raise SystemExit(2) from exc


def analyze_main() -> None:
    from microq_robustness.analysis import ResultsAnalyzer

    parser = argparse.ArgumentParser(description="Analyze results and create paper figures.")
    parser.parse_args()
    try:
        outputs = ResultsAnalyzer(_settings()).run()
        print(f"Analysis complete: {outputs['output_directory']}")
    except RobustnessError as exc:
        LOGGER.error("Analysis failed: %s", exc)
        raise SystemExit(2) from exc

