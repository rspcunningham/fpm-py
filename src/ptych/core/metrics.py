from __future__ import annotations

from typing import TypedDict


# Patch start in capture-space as (y_start, x_start).
type PatchCoord = tuple[int, int]


class InverseMetrics(TypedDict):
    loss: list[float]
    patch_loss: list[list[float]]
    capture_loss: list[list[float]]


class BatchMetricsRecord(TypedDict):
    patches: list[PatchCoord]
    metrics: InverseMetrics
