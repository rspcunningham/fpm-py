# Test 4 Summary

Setup:
- Fixed geometry: `CROP_SIZE=256`, `ROI_SIZE=128`, `UPSAMPLE_RATIO=4`, `tile_batch_size=4`, `N_CAPTURES=61`, `device=mps`.
- Historical reference: old default optimizer at `EPOCHS=1000` from Test 2 on the identical geometry.
- New sweep: shortened solves at `250` epochs, plus a focused `150`-epoch follow-up for the `1e-2` family.

Measured runs:
- `test4_reference_default_1000ep`: schedule `constant`, object LR `0.001`, pupil LR `0.0001`, epochs `1000`, runtime `164.68s`, final loss `0.008135`
- `test4_baseline_const_run1`: schedule `constant`, object LR `0.001`, pupil LR `0.0001`, epochs `250`, runtime `40.53s`, final loss `0.036665`
- `test4_lr3e-3_const_run1`: schedule `constant`, object LR `0.003`, pupil LR `0.0003`, epochs `250`, runtime `40.61s`, final loss `0.009446`
- `test4_lr1e-2_const_run1`: schedule `constant`, object LR `0.01`, pupil LR `0.001`, epochs `250`, runtime `41.15s`, final loss `0.007293`
- `test4_lr3e-3_cosine_run1`: schedule `cosine`, object LR `0.003`, pupil LR `0.0003`, epochs `250`, runtime `41.31s`, final loss `0.018312`
- `test4_lr1e-2_cosine_run1`: schedule `cosine`, object LR `0.01`, pupil LR `0.001`, epochs `250`, runtime `41.49s`, final loss `0.007657`
- `test4_lr1e-2_const_150ep_run1`: schedule `constant`, object LR `0.01`, pupil LR `0.001`, epochs `150`, runtime `24.65s`, final loss `0.007570`
- `test4_lr1e-2_cosine_150ep_run1`: schedule `cosine`, object LR `0.01`, pupil LR `0.001`, epochs `150`, runtime `24.71s`, final loss `0.008903`

Main result:
- Best objective result overall: `test4_lr1e-2_const_run1` at `41.15s` with final loss `0.007293`.
- Best objective result at the more aggressive `150`-epoch budget: `test4_lr1e-2_const_150ep_run1` at `24.65s` with final loss `0.007570`.
- Relative to the old 1000-epoch default, the best 150-epoch setting is `6.68x` faster and `6.9%` lower final loss on this geometry.
- If that speedup carried to the Test 2 full-frame projection, the `CROP_SIZE≈2304` estimate would move from `5.59h` to about `0.84h`.

Interpretation:
- Raising the learning rate helped dramatically once the epoch budget was reduced.
- Cosine decay did not improve the objective loss in this regime; constant-LR runs won at both `250` and `150` epochs for the `1e-2` family.
- The user-observed visual preference for `1e-2 cosine` is still plausible because the objective gap is small and the cosine run can look slightly smoother, but the loss minimum stayed with `1e-2 constant`.
- A strong short-solve default for further scaling tests is `OBJECT_LR=1e-2`, `INTENSITY_LR=1e-2`, `PUPIL_LR=1e-3`, `K_VECTOR_LR=1e-3`, `LR_SCHEDULE="constant"`, and `EPOCHS=150` or `250` depending whether speed or extra optimization margin is the priority.
