"""Cross-field configuration validation before any command mutates artifacts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from microq_phase1.core.config import DegradationConfig, ProjectConfig
from microq_phase1.core.errors import ConfigurationError
from microq_phase1.domain.models import DegradationKind
from microq_phase1.domain.policy import StrictTwoPoolPolicy
from microq_phase1.restorations import RestorationRegistry


class ExperimentConfigurationValidator:
    """Validate parameter schemas, pool identity, and restoration coverage."""

    def __init__(
        self,
        pool_policy: StrictTwoPoolPolicy,
        restorations: RestorationRegistry,
    ) -> None:
        self._pool_policy = pool_policy
        self._restorations = restorations

    def validate(self, config: ProjectConfig) -> None:
        self._pool_policy.validate_pool_configs(config.pools)
        self._validate_recipe_shape(config)
        self._validate_benchmark_binding(config)
        for degradation in config.degradations:
            if degradation.enabled:
                self._validate_degradation(degradation)
        self._validate_restorations(config)

    @staticmethod
    def _validate_recipe_shape(config: ProjectConfig) -> None:
        if config.generation.variants_per_image % 3 != 0:
            raise ConfigurationError(
                "variants_per_image must be divisible by three to preserve severity triplets."
            )

    @staticmethod
    def _validate_benchmark_binding(config: ProjectConfig) -> None:
        benchmark_pools = [pool for pool in config.pools if pool.official_benchmark]
        if len(benchmark_pools) != 1:
            raise ConfigurationError("Exactly one official evaluation benchmark pool is required.")
        if (
            benchmark_pools[0].catalog_path.resolve()
            != config.microbench_export.catalog_path.resolve()
        ):
            raise ConfigurationError(
                "Official pool catalog must match microbench_export.catalog_path."
            )

    def _validate_restorations(self, config: ProjectConfig) -> None:
        rules = config.restoration.rules
        rule_kinds = [rule.degradation for rule in rules]
        if len(rule_kinds) != len(set(rule_kinds)):
            raise ConfigurationError("Restoration rules must be unique per degradation.")
        for rule in rules:
            self._restorations.get(rule.algorithm)
        if not config.restoration.enabled:
            return
        enabled = {item.kind for item in config.degradations if item.enabled}
        missing = enabled - set(rule_kinds)
        if missing:
            raise ConfigurationError(
                f"Missing restoration rules for: {', '.join(sorted(missing))}"
            )

    @staticmethod
    def _validate_degradation(config: DegradationConfig) -> None:
        validator = PARAMETER_VALIDATORS.get(config.kind)
        if validator is None:
            raise ConfigurationError(f"No parameter validator for {config.kind}.")
        for severity, parameters in config.severity_parameters.items():
            try:
                validator(parameters)
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Invalid parameters for {config.kind}/{severity}: {parameters}"
                ) from exc


def _positive(parameters: dict[str, Any], key: str) -> float:
    value = float(parameters[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive.")
    return value


def _fraction(parameters: dict[str, Any], key: str) -> float:
    value = float(parameters[key])
    if value <= 0 or value > 1.0:
        raise ValueError(f"{key} must be in (0, 1].")
    return value


def _sampling(parameters: dict[str, Any]) -> None:
    if float(parameters["factor"]) <= 1:
        raise ValueError("factor must exceed one.")


def _quantization(parameters: dict[str, Any]) -> None:
    if int(parameters["bits"]) not in range(1, 9):
        raise ValueError("bits must be 1..8.")


def _noise(parameters: dict[str, Any]) -> None:
    _positive(parameters, "peak_photons")
    if float(parameters["read_noise_std"]) < 0:
        raise ValueError("read_noise_std must be nonnegative.")


def _defocus(parameters: dict[str, Any]) -> None:
    if _fraction(parameters, "radius_fraction") > 0.25:
        raise ValueError("radius_fraction is implausibly large.")


def _motion(parameters: dict[str, Any]) -> None:
    if _fraction(parameters, "length_fraction") > 0.50:
        raise ValueError("length_fraction is implausibly large.")
    float(parameters["angle_degrees"])


def _periodic(parameters: dict[str, Any]) -> None:
    _positive(parameters, "cycles_per_image")
    float(parameters["angle_degrees"])
    if _fraction(parameters, "amplitude") > 0.50:
        raise ValueError("amplitude is implausibly large.")


def _jpeg(parameters: dict[str, Any]) -> None:
    quality = int(parameters["quality"])
    if quality < 1 or quality > 100:
        raise ValueError("quality must be 1..100.")
    if int(parameters.get("subsampling", 0)) not in {0, 1, 2}:
        raise ValueError("subsampling must be 0, 1, or 2.")


def _photobleaching(parameters: dict[str, Any]) -> None:
    _fraction(parameters, "retained_fraction")
    background = float(parameters.get("background", 0.0))
    if background < 0 or background > 1:
        raise ValueError("background must be in [0, 1].")


def _chromatic(parameters: dict[str, Any]) -> None:
    if _fraction(parameters, "shift_fraction") > 0.25:
        raise ValueError("shift_fraction is implausibly large.")
    float(parameters.get("angle_degrees", 0.0))


def _saturation(parameters: dict[str, Any]) -> None:
    _positive(parameters, "gain")
    black = float(parameters.get("black_level", 0.0))
    if black < 0 or black >= 1:
        raise ValueError("black_level must be in [0, 1).")


PARAMETER_VALIDATORS: dict[DegradationKind, Callable[[dict[str, Any]], None]] = {
    DegradationKind.SAMPLING: _sampling,
    DegradationKind.QUANTIZATION: _quantization,
    DegradationKind.POISSON_GAUSSIAN: _noise,
    DegradationKind.DEFOCUS: _defocus,
    DegradationKind.MOTION: _motion,
    DegradationKind.PERIODIC: _periodic,
    DegradationKind.JPEG: _jpeg,
    DegradationKind.PHOTOBLEACHING: _photobleaching,
    DegradationKind.CHROMATIC_MISREGISTRATION: _chromatic,
    DegradationKind.SATURATION: _saturation,
}
