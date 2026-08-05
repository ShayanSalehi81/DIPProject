"""Validated entities shared across pipeline stages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Compatibility implementation of enum.StrEnum for Python < 3.11."""

        __str__ = str.__str__

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PoolRole(StrEnum):
    TRAINING = "training"
    DEVELOPMENT = "development"
    EVALUATION = "evaluation"


class Modality(StrEnum):
    LIGHT = "light"
    FLUORESCENCE = "fluorescence"
    ELECTRON = "electron"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


class DegradationKind(StrEnum):
    CLEAN = "clean"
    SAMPLING = "sampling"
    QUANTIZATION = "quantization"
    POISSON_GAUSSIAN = "poisson_gaussian"
    DEFOCUS = "defocus"
    MOTION = "motion"
    PERIODIC = "periodic"
    JPEG = "jpeg"
    PHOTOBLEACHING = "photobleaching"
    CHROMATIC_MISREGISTRATION = "chromatic_misregistration"
    SATURATION = "saturation"


class ImageRecord(BaseModel):
    """One clean microscopy image and its provenance."""

    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    image_path: Path
    source_dataset: str = Field(min_length=1)
    license: str = Field(min_length=1)
    pool_role: PoolRole
    modality: Modality
    submodality: str | None = None
    domain: str | None = None
    subdomain: str | None = None
    stain: str | None = None
    microns_per_pixel: float | None = Field(default=None, gt=0)
    mask_path: Path | None = None
    group_id: str | None = None
    is_multichannel: bool = False
    bit_depth: Literal[8, 16] = 8
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_path", "mask_path", mode="before")
    @classmethod
    def normalize_path(cls, value: object) -> object:
        if value is None:
            return None
        # Leave construction of the platform-specific Path subclass to Pydantic.
        # Returning a Path here breaks model_validate_json() under Pydantic 2.10.
        return str(value)


class DegradationStep(BaseModel):
    """One deterministic transformation in a degradation recipe."""

    model_config = ConfigDict(extra="forbid")

    kind: DegradationKind
    severity: Severity
    parameters: dict[str, Any]
    seed: int = Field(ge=0, lt=2**32)


class RestorationStep(BaseModel):
    """One restoration applied in response to a known degradation step."""

    model_config = ConfigDict(extra="forbid")

    degradation: DegradationKind
    algorithm: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class RecipeRecord(BaseModel):
    """Plan for deriving one output image from one clean source image."""

    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    pool_name: str
    pool_role: PoolRole
    source: ImageRecord
    steps: list[DegradationStep]
    output_path: Path
    clean_sha256: str | None = None
    output_sha256: str | None = None
    restoration_steps: list[RestorationStep] = Field(default_factory=list)
    generated: bool = False

    @property
    def is_clean_control(self) -> bool:
        return not self.steps


class AnalysisRecord(BaseModel):
    """Computed DIP evidence and full-reference measurements for one recipe."""

    recipe_id: str
    source_image_id: str
    pool_role: PoolRole
    modality: Modality
    source_dataset: str
    degradation: str
    severity: str
    step_count: int
    restoration_algorithms: list[str] = Field(default_factory=list)
    features: dict[str, float]
    quality_metrics: dict[str, float]
    morphology_metrics: dict[str, float] = Field(default_factory=dict)
