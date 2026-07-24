"""Typed exceptions presented consistently by the command-line interface."""


class MicroQError(Exception):
    """Base class for expected project errors."""


class ConfigurationError(MicroQError):
    """Raised when a project configuration is unsafe or incomplete."""


class CatalogError(MicroQError):
    """Raised when a source catalog is malformed or inconsistent."""


class PoolPolicyError(MicroQError):
    """Raised when strict train/evaluation separation is violated."""


class DegradationError(MicroQError):
    """Raised when a degradation cannot be applied safely."""


class RestorationError(MicroQError):
    """Raised when a restoration recipe is invalid."""


class ValidationFailure(MicroQError):
    """Raised when an artifact fails a required quality gate."""

