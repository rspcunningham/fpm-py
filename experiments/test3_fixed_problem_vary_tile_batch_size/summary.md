# Test 3 Summary

Measured settings: `CROP_SIZE=512`, `ROI_SIZE=128`, `N_CAPTURES=61`, `UPSAMPLE_RATIO=4`, `EPOCHS=1000`, `device=mps`.

## Measured runs

- `tile_batch_size=1` (16 batches): total runtime `1290.24s`, runtime/epoch `0.080640s`, final loss `0.005520`.
- `tile_batch_size=2` (8 batches): total runtime `783.55s`, runtime/epoch `0.097944s`, final loss `0.006150`.
- `tile_batch_size=4` (4 batches): total runtime `970.81s`, runtime/epoch `0.242702s`, final loss `0.007514`.
- `tile_batch_size=8` (2 batches): total runtime `624.79s`, runtime/epoch `0.312393s`, final loss `0.007843`.

## Main result

- Fastest total runtime: `tile_batch_size=8` at `624.79s`.
- Best final loss: `tile_batch_size=1` at `0.005520`.
- Relative to `tile_batch_size=1`, the fastest condition is `2.07x` faster in total runtime.

## Interpretation

- Increasing `tile_batch_size` reduced the number of batch solves from `16` down to `2`, and the fastest measured condition was `tile_batch_size=8`.
- Lower `runtime_per_epoch_sec` at small batch sizes did not translate into lower total runtime, because total runtime is dominated by repeated batch solves.
- Final loss worsened as `tile_batch_size` increased in this sweep, so throughput and reconstruction quality were not aligned on the same optimum.
- The `tile_batch_size=4` point was reused from Test 2 with identical geometry rather than rerun in this sweep; interpret small differences around that point with caution.

## Observational note

- During the sweep, system monitoring showed low CPU usage, no swap, negligible disk activity, and GPU usage pinned near saturation.
- The run logs showed bursty throughput with occasional long stalls, which is consistent with a GPU/backend-limited execution regime rather than a CPU or I/O bottleneck.
