"""Best-effort conversion of nested polygon annotations into binary PNG masks."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


class PolygonMaskExporter:
    """Rasterize common flat and point-pair polygon representations."""

    def export(
        self,
        polygon: object,
        *,
        width: int,
        height: int,
        output_path: Path,
    ) -> Path | None:
        rings = list(self._rings(polygon))
        if not rings:
            return None
        canvas = Image.new("L", (width, height), color=0)
        drawer = ImageDraw.Draw(canvas)
        drawn = 0
        for ring in rings:
            scaled = self._scale_if_normalized(ring, width, height)
            if len(scaled) < 3:
                continue
            drawer.polygon(scaled, fill=255)
            drawn += 1
        if drawn == 0:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        canvas.save(temporary, format="PNG")
        os.replace(temporary, output_path)
        return output_path.resolve()

    def _rings(self, value: object) -> Iterator[list[tuple[float, float]]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return
        items = list(value)
        if self._is_flat_ring(items):
            yield [
                (float(items[index]), float(items[index + 1]))
                for index in range(0, len(items), 2)
            ]
            return
        if self._is_point_ring(items):
            yield [(float(point[0]), float(point[1])) for point in items]
            return
        for child in items:
            yield from self._rings(child)

    @staticmethod
    def _is_flat_ring(items: list[Any]) -> bool:
        return (
            len(items) >= 6
            and len(items) % 2 == 0
            and all(isinstance(value, Real) for value in items)
        )

    @staticmethod
    def _is_point_ring(items: list[Any]) -> bool:
        return len(items) >= 3 and all(
            isinstance(point, Sequence)
            and not isinstance(point, (str, bytes))
            and len(point) == 2
            and all(isinstance(value, Real) for value in point)
            for point in items
        )

    @staticmethod
    def _scale_if_normalized(
        ring: list[tuple[float, float]],
        width: int,
        height: int,
    ) -> list[tuple[float, float]]:
        maximum = max(max(abs(x), abs(y)) for x, y in ring)
        if maximum > 1.5:
            return ring
        return [(x * (width - 1), y * (height - 1)) for x, y in ring]
