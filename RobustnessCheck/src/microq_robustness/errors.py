"""Typed errors surfaced by the Phase 1.5 command-line programs."""


class RobustnessError(Exception):
    """Base class for expected robustness-check failures."""


class ConfigurationError(RobustnessError):
    """Raised when environment configuration is invalid."""


class ManifestError(RobustnessError):
    """Raised when Phase 1 manifests cannot support a paired study."""


class ServerError(RobustnessError):
    """Raised when the vLLM endpoint is unavailable or incompatible."""


class AnalysisError(RobustnessError):
    """Raised when evaluation results are incomplete or malformed."""

