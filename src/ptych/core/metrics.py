from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import torch

from ptych.core.pupil import Pupil


# Tile start in capture-space as (y_start, x_start).
type TileCoord = tuple[int, int]


class InverseMetrics(TypedDict):
    loss: list[float]
    tile_loss: list[list[float]]
    capture_loss: list[list[float]]


class BatchMetricsRecord(TypedDict):
    tiles: list[TileCoord]
    metrics: InverseMetrics


type TileCompleteCallback = Callable[
    [int, int, torch.Tensor, Pupil, InverseMetrics], None
]
type BatchCompleteCallback = Callable[
    [list[TileCoord], list[Pupil], InverseMetrics], None
]
