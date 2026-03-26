Test 1: Fixed field of view, vary tile size

- Keep `CROP_SIZE` fixed.
- Keep `EPOCHS`, `N_CAPTURES`, `UPSAMPLE_RATIO`, `tile_batch_size`, device, and pupil settings fixed.
- Vary `ROI_SIZE` so the same cropped field of view is partitioned into different numbers of tiles.
- Example: if `CROP_SIZE = 256`, use `ROI_SIZE in {256, 128, 64}` to get `1`, `4`, and `16` tiles.
- Measure total runtime.
- Measure runtime per epoch.
- Record final loss.

Purpose:
This isolates the effect of chopping the same reconstructed area into more, smaller tiles.


Test 2: Fixed tile size, vary field of view

- Keep `ROI_SIZE` fixed.
- Keep `EPOCHS`, `N_CAPTURES`, `UPSAMPLE_RATIO`, `tile_batch_size`, device, and pupil settings fixed.
- Vary `CROP_SIZE` to increase the total reconstructed field of view.
- Choose `CROP_SIZE` values that are exact multiples of `ROI_SIZE`.
- Measure total runtime.
- Measure runtime per epoch.
- Record final loss.

Purpose:
This measures how runtime scales as the total reconstructed area increases while tile granularity stays fixed.


Test 3: Fixed reconstruction problem, vary tile batch size

- Keep `CROP_SIZE` fixed.
- Keep `ROI_SIZE` fixed.
- Keep `EPOCHS`, `N_CAPTURES`, `UPSAMPLE_RATIO`, device, and pupil settings fixed.
- Vary `tile_batch_size`.
- Example: use `tile_batch_size in {1, 2, 4}`.
- Measure total runtime.
- Measure runtime per epoch.
- Record final loss.

Purpose:
This tests whether the current runtime is limited by batch shape or memory pressure rather than only by tile count or field of view.


Test 4: Higher learning rate, fewer epochs

- Keep the reconstruction setup fixed.
- Reduce `EPOCHS` and increase the optimizer learning rates in `src/ptych/core/inverse.py`.
- Keep the object and pupil parameter groups separate rather than forcing them to share one learning rate.
- Compare final loss at a fixed reduced epoch budget.
- Also compare runtime and inspect reconstruction quality qualitatively.

Suggested approach:

- Start from the current defaults.
- Try a small sweep of larger learning rates for the object parameters, for example `1e-3`, `3e-3`, `1e-2`.
- Keep the pupil learning rate lower than the object learning rate, for example by about `10x`.
- If loss becomes noisy or diverges, back off.

How to find a good learning rate:

- Use a simple learning-rate range test on a representative reconstruction setting.
- Increase the learning rate across a short run and watch where the loss stops improving cleanly or becomes unstable.
- Choose a learning rate below that instability point.

Purpose:
This tests whether a more aggressive optimizer setup can recover similar quality in fewer epochs, reducing total runtime without changing the reconstruction geometry.


Notes

- Run each condition at least twice and compare the second run to reduce one-time warmup effects.
- Use the same dataset and capture subset for all conditions.
- If possible, also record peak memory usage, since memory pressure may affect runtime on MPS.
