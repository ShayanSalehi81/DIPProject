from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microq_phase1.domain.models import ImageRecord, Modality, PoolRole
from microq_phase1.domain.strategies import ImageFrame


@pytest.fixture
def fluorescence_record(tmp_path: Path) -> ImageRecord:
    return ImageRecord(
        image_id="fluorescence-1",
        image_path=tmp_path / "image.png",
        source_dataset="external-source",
        license="CC-BY-4.0",
        pool_role=PoolRole.TRAINING,
        modality=Modality.FLUORESCENCE,
        submodality="confocal microscopy",
        group_id="field-1",
        is_multichannel=True,
    )


@pytest.fixture
def color_frame(fluorescence_record: ImageRecord) -> ImageFrame:
    gradient = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    gray = np.tile(gradient, (64, 1))
    pixels = np.stack([gray, np.flipud(gray), np.fliplr(gray)], axis=-1)
    return ImageFrame(pixels=pixels.astype(np.float32), record=fluorescence_record)

