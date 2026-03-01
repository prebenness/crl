# Basic vs Shared Broad Sweeps

## Scope

Ran broad coarse sweeps for ANBN under one fixed schedule family to compare how wide the viable hyperparameter region is for basic MDL vs corrected shared MDL.

Fixed schedule for all runs:
- `epochs = 900`
- `warmup_epochs = 600`
- `lr = 0.03`
- `n_max = 10`, `m_max = 10`
- `n_samples = 32`
- `tau_start = 1.0`, `tau_end = 0.02`
- full-batch training
- `seed = 0`, `data_seed = 0`

## Basic MDL Sweep (`mdl_lambda`)

Sweep values:
- `0.003, 0.01, 0.03, 0.1, 0.144, 0.3, 1.0`

Short-range (`test_max_n = 5000`) results:
- `0.003`: `1357` bits, `gen_n = 1`
- `0.01`: `1345` bits, `gen_n = 1`
- `0.03`: `1269` bits, `gen_n = 1`
- `0.1`: `947` bits, `gen_n = 5000`
- `0.144`: `783` bits, pathological (`n_perfect = 4999`, but not contiguous)
- `0.3`: `779` bits, `gen_n = 1173`
- `1.0`: `577` bits, collapse

Direct `n = 100000` checks on representative points:
- `0.1`: `947` bits, perfect through `100000`
- `0.144`: `783` bits, fails immediately (`gen_n = 0`)
- `0.3`: `779` bits, only `gen_n = 1161`

## Shared MDL Sweep (`lambda1`, corrected objective)

Sweep values for `lambda1`:
- `0.01, 0.03, 0.1, 0.144, 0.3, 1.0`

Anchors tested for `lambda2`:
- `20`
- `100`

### `lambda2 = 20`

Short-range (`test_max_n = 5000`) results:
- `0.01`: `1351` bits, `gen_n = 1`
- `0.03`: `1287` bits, `gen_n = 1`
- `0.1`: `1011` bits, `gen_n = 1`
- `0.144`: `945` bits, `gen_n = 5000`
- `0.3`: `729` bits, `gen_n = 5000`
- `1.0`: `811` bits, `gen_n = 5000`

Direct larger-`n` checks:
- `0.144`: perfect through `100000` in the sweep run, and separately verified to `1e6` earlier
- `0.3`: `729` bits, perfect through `100000`
- `1.0`: `811` bits, perfect through `100000`

### `lambda2 = 100`

Short-range (`test_max_n = 5000`) results:
- `0.01`: `1347` bits, `gen_n = 1`
- `0.03`: `1273` bits, `gen_n = 1`
- `0.1`: `979` bits, `gen_n = 5000`
- `0.144`: `907` bits, `gen_n = 5000`
- `0.3`: `709` bits, `gen_n = 5000`
- `1.0`: `581` bits, collapse

Direct larger-`n` checks:
- `0.1`: `979` bits, perfect through `100000`
- `0.144`: perfect through `100000` in the sweep run, and separately verified to `1e6` earlier
- `0.3`: `709` bits, perfect through `100000`

## Main Conclusion

Under this fixed schedule family, corrected shared MDL is materially more stable across regularization strength than basic MDL.

- Basic MDL only has one clearly robust point in this sweep (`mdl_lambda = 0.1`).
- Nearby values can look compressed but be pathological (`0.144`) or degrade sharply (`0.3`).
- Corrected shared MDL has a wider viable band once `lambda2` is large enough:
  - at `lambda2 = 20`, strong behavior from `lambda1 ≈ 0.144` up to `1.0`
  - at `lambda2 = 100`, strong behavior from `lambda1 ≈ 0.1` up to `0.3`

This does not mean shared MDL is better overall yet:
- best basic frontier remains much more compressed (`701` bits through `1e6`)
- but for stability under coarse hyperparameter variation, shared currently looks better in this schedule family.
