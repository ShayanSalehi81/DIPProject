from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from microq_phase1.infrastructure.polygons import PolygonMaskExporter


def test_polygon_exporter_accepts_normalized_flat_coordinates(tmp_path: Path) -> None:
    output = tmp_path / "mask.png"
    result = PolygonMaskExporter().export(
        [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
        width=32,
        height=32,
        output_path=output,
    )
    assert result == output.resolve()
    with Image.open(output) as mask:
        pixels = np.asarray(mask)
    assert int((pixels > 0).sum()) > 0
