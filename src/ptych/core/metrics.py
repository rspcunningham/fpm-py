from __future__ import annotations

from typing import TypedDict


# Tile start in capture-space as (y_start, x_start).
type TileCoord = tuple[int, int]


class InverseMetrics(TypedDict):
    loss: list[float]
    tile_loss: list[list[float]]
    capture_loss: list[list[float]]


class BatchMetricsRecord(TypedDict):
    tiles: list[TileCoord]
    metrics: InverseMetrics
