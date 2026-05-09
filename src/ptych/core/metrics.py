from __future__ import annotations

from typing import TypedDict


# Patch start in capture-space as (y_start, x_start).
type PatchCoord = tuple[int, int]


class InverseRunSummary(TypedDict):
    elapsed_seconds: float
    losses_finite: bool
    final_loss: float
    best_loss: float


class InverseMetrics(TypedDict):
    loss: list[float]
    patch_loss: list[list[float]]
    illumination_loss: list[list[float]]
    summary: InverseRunSummary


class BatchMetricsRecord(TypedDict):
    patches: list[PatchCoord]
    metrics: InverseMetrics
