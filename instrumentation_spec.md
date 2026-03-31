# Reconstruction Instrumentation Spec

This document proposes a concrete instrumentation plan for understanding why some reconstructions succeed and others fail in the current Fourier ptychography pipeline.

It is written against the current entrypoint and solver stack:

- [`reconstruction_demo.py`](/Users/robin/Desktop/parasight/fpm-py/reconstruction_demo.py)
- [`src/ptych/reconstruct.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/reconstruct.py)
- [`src/ptych/core/tiled.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/tiled.py)
- [`src/ptych/core/inverse.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/inverse.py)
- [`src/ptych/core/forward.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/forward.py)
- [`src/ptych/core/metrics.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/metrics.py)
- [`preview_utils.py`](/Users/robin/Desktop/parasight/fpm-py/preview_utils.py)

The goal is not to add more generic logging. The goal is to make failures attributable to one of a small number of causes:

- weak or biased capture data
- insufficient Fourier coverage
- poor initialization
- optimizer instability or compensation behavior
- pupil or illumination mismatch
- local tile ambiguity
- stitching and seam artifacts

## Why The Current Telemetry Is Not Enough

Today the reconstruction path exposes only:

- total loss over epoch
- per-tile loss over epoch
- final per-capture loss
- final stitched object preview

That is useful for tracking optimization progress, but not for explaining failure. In particular, the current metrics do not show:

- what the selected captures actually look like after demosaic, crop, and normalization
- which capture geometries are informative or redundant
- whether the pupil is compensating for object error
- whether the model is recovering high frequencies or only reducing low-frequency photometric error
- whether a few captures dominate the failure
- whether tile boundaries or tile-specific pupil drift are responsible for visible artifacts

## Design Principles

The instrumentation should follow four rules.

1. Record dense scalar telemetry, but sparse image telemetry.
   Scalar traces can be kept every epoch. Image artifacts should be emitted at checkpoints only.

2. Keep the trace tied to the actual solver hierarchy.
   Instrument input preparation, batch solve, tile solve, checkpoint solve, and stitch output separately.

3. Support both real-data and synthetic-truth modes.
   Synthetic runs are the cleanest way to separate optimizer failure from data/physics failure.

4. Make the default artifacts easy to inspect offline.
   A saved run directory with JSON/NPY/PNG plus a simple HTML summary is enough.

## Questions The Instrumentation Must Answer

The spec should make the following questions answerable from one run directory.

- Which captures were selected, and how much Fourier support do they provide?
- Are the hard captures associated with certain illumination angles, wavelengths, or low-SNR regions?
- Does the object recover stable high frequencies over training, or only reduce low-frequency loss?
- Does the learned pupil stay physically plausible and consistent across tiles?
- Are failures localized to certain crop regions or tile seams?
- In synthetic mode, is the error driven by missing data, optimizer limitations, or model mismatch?

## Instrumentation Layers

### 1. Input Diagnostics

Attach this at study preparation time in [`src/ptych/reconstruct.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/reconstruct.py).

Record:

- run config:
  - dataset id or local path
  - `n_captures`
  - `capture_region`
  - `tile_size`
  - `object_to_capture_ratio`
  - `epochs`
  - `tile_batch_size`
  - optimizer learning rates if configurable
- selected capture metadata per capture:
  - original capture index
  - filename
  - wavelength
  - selected RGB channel index
  - raw `(kx, ky)` on camera grid
  - renormalized `(kx, ky)` on object grid
  - illumination radius and angle
  - exposure if present
- selected capture statistics after demosaic and crop:
  - min, max, mean, std
  - percentile summary
  - fraction near zero
  - fraction near saturation
  - center-of-mass / simple asymmetry statistics
- global normalization metadata:
  - scalar max used for normalization
  - whether a few captures dominate that max

Artifacts:

- `run_config.json`
- `selected_captures.csv`
- `capture_contact_sheet.png`
- `k_space_coverage.png`

Why it matters:

- bad runs may be caused upstream by an unhelpful crop, weak dynamic range, or poor illumination coverage before the inverse solver even starts

### 2. Optimization Trace

Attach this inside [`src/ptych/core/inverse.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/inverse.py).

Record every epoch:

- existing loss terms:
  - total loss
  - per-tile loss
  - per-capture loss
- parameter-state summaries:
  - object amplitude mean/std/min/max per tile
  - object phase mean/std per tile
  - `intensity_scale` values per capture
  - phase-coefficient norm per tile
  - amplitude-coefficient norm per tile
  - `rad_fraction` per tile
- update and stability summaries:
  - gradient norm for object amplitude
  - gradient norm for object phase
  - gradient norm for intensity scale
  - gradient norm for pupil params
  - parameter delta norm relative to previous checkpoint

Record at sparse checkpoints only:

- object amplitude image per tile
- object phase image per tile
- selected predicted captures
- measured captures for the same probe set
- residual maps for the same probe set

Why it matters:

- this distinguishes healthy convergence from compensation behavior such as the pupil or `intensity_scale` absorbing the mismatch

### 3. Fourier-Space Probes

Attach this where full-resolution predicted intensities and object estimates already exist in [`src/ptych/core/inverse.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/inverse.py) and [`src/ptych/core/forward.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/forward.py).

Record at checkpoints:

- object Fourier magnitude per tile
- radial power spectrum per tile
- high-frequency power fraction per tile
- Fourier residual map for probe captures
- union-of-shifted-pupil support mask implied by the selected `(kx, ky)` and current pupil radius

Derived summaries:

- frequency recovery over training
- missing-frequency regions
- anisotropy of recovered detail

Why it matters:

- many visually plausible reconstructions are actually only low-frequency fits
- this layer shows whether extra detail is real, missing, or hallucination-like compensation

### 4. Pupil And Illumination Diagnostics

Attach this in [`src/ptych/core/inverse.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/inverse.py) and after tile solve in [`src/ptych/core/tiled.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/tiled.py).

Record:

- pupil amplitude and phase images per tile at final solve
- pupil amplitude and phase at checkpoints for one or two representative tiles
- Zernike coefficient trajectories
- `rad_fraction` trajectory
- if `learn_k_vectors=True`, k-vector trajectory and final offset from initialization

Cross-tile consistency summaries:

- coefficient variance across tiles
- pupil-image similarity between neighboring tiles
- radius drift across tiles

Why it matters:

- if one image reconstructs badly because the solver pushes the pupil into inconsistent local optima, that should be visible immediately

### 5. Tile And Stitch Diagnostics

Attach this in [`src/ptych/core/tiled.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/tiled.py).

Record:

- tile plan metadata:
  - tile start
  - trim region
  - owned output region
- final loss per tile
- tile difficulty map over the crop
- overlap disagreement before trimming
- seam-energy map after stitch
- neighboring tile amplitude and phase discontinuity scores

Artifacts:

- `tile_layout.png`
- `tile_difficulty_heatmap.png`
- `seam_energy_map.png`

Why it matters:

- some images will “fail” primarily at seams or in low-information local regions, and that is a tiled-reconstruction problem rather than a core inverse-model problem

### 6. Synthetic Ground-Truth Diagnostics

Use the synthetic pipeline via [`synthetic_demo.py`](/Users/robin/Desktop/parasight/fpm-py/synthetic_demo.py) and [`src/ptych/core/synthetic.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/synthetic.py).

When the input is synthetic, record all the previous telemetry plus:

- object amplitude error against truth
- object phase error against truth
- pupil error against truth
- capture prediction error against truth
- frequency recovery against truth

Recommended controlled sweeps:

- object class:
  - sparse edges
  - repeated texture
  - low-contrast texture
  - phase-heavy object
- measurement quality:
  - Poisson-like noise
  - additive read noise
  - clipping / saturation
  - exposure scaling mismatch
- geometry:
  - reduced capture count
  - biased illumination angles
  - radius truncation
  - crop changes
- model mismatch:
  - wrong NA
  - wrong pupil radius
  - wrong k-vectors
  - demosaic/channel mismatch

Why it matters:

- synthetic truth is the cleanest way to map observed artifacts to specific causal failures

## Proposed Trace Structure

The current [`InverseMetrics`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/metrics.py#L16) is too small for this use case. Conceptually, the trace should split into scalar histories, checkpoint artifacts, and final summaries.

Suggested top-level structure:

```text
RunTrace
  config
  input_diagnostics
  batches[]
    batch_id
    tile_coords[]
    scalar_history
    checkpoints[]
    final_tile_summaries[]
  stitched_summary
  synthetic_truth_summary?  # optional
```

Suggested scalar history fields:

- `epoch`
- `total_loss`
- `tile_loss[T]`
- `capture_loss[B]`
- `intensity_scale[B]`
- `grad_norms`
- `object_norms[T]`
- `pupil_norms[T]`
- `rad_fraction[T]`
- `high_frequency_power[T]`

Suggested checkpoint artifact bundle:

- object amplitude image per tile
- object phase image per tile
- pupil amplitude and phase image per tile
- selected measured captures
- selected predicted captures
- selected residual maps
- object Fourier magnitude
- frequency-support overlay

The important design choice is to keep large tensors off the scalar JSON path. Save arrays as `.npy` and render small `.png` previews for inspection.

## Callback Shape

The current callback types in [`src/ptych/core/metrics.py`](/Users/robin/Desktop/parasight/fpm-py/src/ptych/core/metrics.py) are too narrow for rich probing. Conceptually, the callback surface should be expanded in layers rather than overloaded into one callback.

Recommended callback families:

- `on_input_prepared(input_trace)`
- `on_epoch(epoch_trace)`
- `on_checkpoint(checkpoint_trace)`
- `on_batch_complete(batch_trace)`
- `on_tile_complete(tile_trace)`
- `on_stitch_complete(stitch_trace)`

Suggested payload contents:

### `input_trace`

- run config
- capture metadata table
- basic capture statistics
- coverage summary

### `epoch_trace`

- scalar-only values for the current epoch
- no large images

### `checkpoint_trace`

- epoch number
- scalar snapshot
- sparse tensors or file handles for object, pupil, predictions, residuals, and spectra

### `batch_trace`

- tile coordinates in the batch
- final scalar history for that batch
- batch-level hardest captures
- batch-level hardest tiles

### `tile_trace`

- tile coordinate
- final object summary
- final pupil summary
- tile-local probe results

### `stitch_trace`

- seam metrics
- tile difficulty map
- final stitched preview summary

This structure keeps the cost predictable and makes it easy to disable expensive probes.

## Probe Sets

Not every capture should be rendered at every checkpoint. Use a fixed small probe set.

Recommended default probe captures:

- on-axis or nearest-to-center illumination
- one medium-angle illumination
- one high-angle illumination near the edge of support
- one capture with historically high loss
- one capture with historically low loss

If the probe set changes dynamically, save the selection policy and selected indices.

## Dashboard Layout

The first useful dashboard does not need to be interactive. A simple HTML page or report image set is enough.

Recommended sections:

### 1. Run Summary

- dataset and config
- crop and tile geometry
- final scalar summaries
- wall-clock timing

### 2. Input View

- capture contact sheet
- `(kx, ky)` scatter plot
- Fourier support / coverage map
- per-capture brightness and dynamic-range chart

### 3. Optimization View

- total loss over epoch
- hardest-capture loss traces
- parameter trajectories:
  - `intensity_scale`
  - `rad_fraction`
  - coefficient norms
  - gradient norms

### 4. Probe Capture View

For each probe capture:

- measured capture
- predicted capture
- residual image
- residual histogram

Show the first checkpoint, a middle checkpoint, and the final checkpoint.

### 5. Object Recovery View

- object amplitude checkpoints
- object phase checkpoints
- Fourier magnitude checkpoints
- radial power spectrum over time

### 6. Pupil View

- initial vs final pupil for representative tiles
- Zernike trajectories
- cross-tile pupil consistency summary

### 7. Tile And Stitch View

- tile layout
- per-tile final loss heatmap
- seam-energy map
- final stitched amplitude and phase previews

### 8. Synthetic Truth View

Only for synthetic runs:

- truth object vs reconstruction
- truth pupil vs learned pupil
- truth error maps

## Minimal Viable Instrumentation

If the goal is to move quickly and get immediate explanatory value, the first pass should not attempt everything above.

Recommended phase 1:

- input diagnostics:
  - capture metadata table
  - `(kx, ky)` plot
  - capture contact sheet
- optimization trace:
  - total loss
  - per-capture loss over epoch
  - `intensity_scale`
  - `rad_fraction`
  - Zernike coefficient norms
- checkpoint artifacts every fixed interval:
  - object amplitude
  - object phase
  - measured/predicted/residual for 3 to 5 probe captures
  - Fourier magnitude
- tile summary:
  - per-tile final loss
  - seam-energy map

This phase alone should already explain a large fraction of bad reconstructions.

Recommended phase 2:

- gradient norms
- cross-tile pupil consistency
- overlap disagreement diagnostics
- frequency-support overlays

Recommended phase 3:

- synthetic truth sweeps
- capture ablation and influence analysis
- optional interactive dashboard

## Performance And Storage Constraints

Instrumentation can distort the runtime if it is not controlled.

Guardrails:

- save scalar histories every epoch
- save image checkpoints every `N` epochs only
- render only a fixed probe set of captures
- save large tensors as `.npy`, not embedded JSON
- keep preview PNGs downsampled for browsing
- allow instrumentation levels such as:
  - `off`
  - `light`
  - `standard`
  - `full`

Suggested meaning:

- `light`: scalars only
- `standard`: scalars plus sparse checkpoint images
- `full`: full checkpoint artifacts plus synthetic-truth extras

## Failure Taxonomy This Spec Supports

Once the above is in place, bad runs can be classified more cleanly.

### Data-limited

Symptoms:

- weak or uneven capture statistics
- poor k-space coverage
- a few captures dominate normalization
- hard captures cluster at specific illumination angles

### Optimization-limited

Symptoms:

- total loss plateaus early
- gradient norms collapse or explode
- probe residuals stop improving even though some scalars drift

### Compensation-limited

Symptoms:

- `intensity_scale` drifts strongly
- pupil coefficients grow while object detail does not improve
- predicted captures match brightness but miss structure

### Frequency-limited

Symptoms:

- low loss but weak high-frequency power growth
- radial spectrum fails to fill outer bands
- residual energy concentrates in high-frequency regions

### Tile-limited

Symptoms:

- loss concentrates in certain tiles
- neighboring tiles learn inconsistent pupils
- seam-energy map is strong even when local tile losses are low

## Recommended First Success Criteria

The instrumentation is useful when a single run can support statements like:

- "This failed because the chosen crop had low signal in the high-angle captures."
- "This converged in loss, but the Fourier spectrum never recovered outer-band detail."
- "The pupil drifted differently in edge tiles, which is why the stitched seams are visible."
- "Only three captures are persistently hard, and they all sit at one illumination quadrant."
- "On synthetic truth, the optimizer is fine; the real failure is model mismatch or data preprocessing."

## Suggested Output Layout

One practical run directory layout:

```text
results/<run_name>/
  run_config.json
  selected_captures.csv
  scalar_history.json
  input/
    capture_contact_sheet.png
    k_space_coverage.png
  checkpoints/
    epoch_0000/
    epoch_0050/
    epoch_0100/
  tiles/
    tile_summary.csv
    tile_difficulty_heatmap.png
    seam_energy_map.png
  stitched/
    stitched_object.npy
    stitched_amplitude.png
    stitched_phase.png
  report/
    index.html
```

## Final Recommendation

The most important change is architectural, not cosmetic:

- treat instrumentation as a first-class trace pipeline
- expand metrics beyond scalar loss
- separate scalar histories from sparse checkpoint artifacts
- support both real-data and synthetic-truth runs with the same reporting shape

If the implementation follows that structure, the repo will be able to answer not only whether a reconstruction is good, but why it is good, why it failed, and which part of the pipeline is responsible.
