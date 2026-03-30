# FPM-Py

Fourier ptychography in PyTorch.

This repository contains a forward model, an inverse solver, dataset loading utilities, and a pair of runnable demos for synthetic data generation and tiled reconstruction. The current codebase is centered around study manifests (`info.json` + `captures/`) and a tiled reconstruction pipeline built on PyTorch tensors.

## What It Does

- Loads Fourier ptychography studies from disk or from the configured dataset cache
- Computes illumination k-vectors from LED geometry and optics metadata
- Simulates capture stacks from a complex object and pupil
- Reconstructs a higher-resolution complex object from measured captures
- Supports tiled reconstruction for larger fields of view
- Produces previews and simple experiment artifacts under `results/` and `experiments/`

## Quick Start

### Requirements

- Python `>=3.13,<3.14`
- [`uv`](https://docs.astral.sh/uv/)

Install dependencies:

```bash
uv sync
```

Load a study either by local path or by dataset name:

```python
from ptych import PtychStudy

local_study = PtychStudy.load("results/synthetic_usaf_test")
cached_or_remote_study = PtychStudy.load("usaf-test")
```

## Demos

### Synthetic Data Demo

```bash
uv run synthetic_demo.py
```

What it does:

- Loads the `usaf-test` study manifest for optics and illumination geometry
- Reads [`demo_images/ideal.png`](demo_images/ideal.png) as the ideal object amplitude
- Builds a complex object and an ideal pupil
- Simulates captures with the shared forward model
- Writes a synthetic study to `results/synthetic_usaf_test/`

You can reconstruct that generated study with:

```python
from ptych import PtychStudy

study = PtychStudy.load("results/synthetic_usaf_test")
```

Output layout:

- `results/synthetic_usaf_test/info.json`
- `results/synthetic_usaf_test/captures/*.npy`

### Reconstruction Demo

```bash
uv run reconstruction_demo.py
```

What it does:

- Loads the `usaf-test` dataset through [`PtychStudy`](src/ptych/data/study.py)
- Crops and demosaics the capture stack
- Initializes an ideal pupil parameterization
- Reconstructs tiled high-resolution complex objects with the inverse solver
- Stitches the reconstructed tiles into a single output

Output layout:

- `results/reconstruction_usaf_test/reconstruction_metrics.png`
- `results/reconstruction_usaf_test/stitched_object.npy`
- `results/reconstruction_usaf_test/stitched_object.png`

## Data Model

A study consists of:

- `info.json`: acquisition metadata and per-capture illumination metadata
- `captures/`: `.npy` intensity images referenced by the manifest

The manifest schema is documented in [INFO_JSON_SCHEMA.md](INFO_JSON_SCHEMA.md).

At runtime, the main container is [`PtychStudy`](src/ptych/data/study.py), which holds:

- `manifest`
- `captures`
- `kx_batch`
- `ky_batch`

`kx_batch` and `ky_batch` are computed from LED positions and normalized to the camera grid before being converted to the object grid inside the synthetic and inverse paths.

## Core Concepts

### Object-To-Capture Ratio

The codebase uses `object_to_capture_ratio` for the linear scale factor between the high-resolution object grid and the lower-resolution capture grid:

```text
object_to_capture_ratio = N / n
```

This ratio is used to:

- size the reconstructed object grid
- convert camera-grid-normalized `kx/ky` to object-grid normalization
- average-pool model predictions back to the capture grid
- define the effective object-plane sampling used to build the pupil

### Tiled Reconstruction

The reconstruction pipeline solves square capture tiles independently, then stitches the reconstructed high-resolution tiles into one output image. The main entry point is [`solve_study`](src/ptych/reconstruct.py), which delegates to the tiled solver in [`src/ptych/core/tiled.py`](src/ptych/core/tiled.py).

## Public API

The package currently exports:

- [`PtychStudy`](src/ptych/data/study.py)
- [`forward_model`](src/ptych/core/forward.py)
- [`solve_inverse`](src/ptych/core/inverse.py)
- [`solve_study`](src/ptych/reconstruct.py)
- [`StudySolveResult`](src/ptych/reconstruct.py)

## Repository Layout

```text
src/ptych/
  core/
    forward.py      Shared forward model
    inverse.py      Inverse solver
    pupil.py        Pupil parameterization and Zernike helpers
    synthetic.py    Synthetic capture generation
    tiled.py        Tiled reconstruction and stitching
  data/
    study.py        Study container and loading
    synthetic.py    Synthetic study writer
    utils.py        k-vector computation and capture preparation
    download/       Dataset cache and transport
reconstruction_demo.py
synthetic_demo.py
preview_utils.py
experiments/
demo_images/
```

## Dataset Access

`PtychStudy.load(...)` accepts either:

- a local study directory path containing `info.json` and `captures/`
- a dataset name such as `usaf-test`

For dataset names, the loader resolves through the local cache in `~/.cache/ptych/datasets` and downloads missing datasets through the configured Nextcloud transport.

The demos currently use:

- `usaf-test`

## Current Assumptions And Limitations

- Reconstruction currently assumes square object tensors in the synthetic path
- `prepare_captures()` requires exactly one LED per non-darkfield capture
- Mixed wavelengths within a single loaded study are not supported by the current loader path
- Darkfield captures are represented in the schema but skipped by the current prepared-input pipeline
- The demos are research-oriented scripts, not polished CLI commands

## Experiments

The [`experiments/`](experiments) directory contains saved runs, comparisons, and summaries for reconstruction sweeps such as:

- field-of-view vs tile size
- crop size vs runtime/output
- tile batch size
- learning-rate schedules
- object-to-capture ratio

## Development

Common commands:

```bash
uv sync
uv run synthetic_demo.py
uv run reconstruction_demo.py
uv run python -m compileall src reconstruction_demo.py synthetic_demo.py
```

## License

MIT. See [LICENSE](LICENSE).
