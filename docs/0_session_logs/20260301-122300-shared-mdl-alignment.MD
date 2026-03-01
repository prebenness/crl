# Shared MDL Alignment

## Scope

Aligned the shared-weight MDL implementation with the revised proposal.

## Changes

1. Updated the shared coding term from `sum_i KL(pi_i || phi)` to `sum_i CE_2(pi_i, phi)` in `src/mdl/shared_weights.py`.
2. Updated `P_base` to use the Lan-style codelength prior: `P_base(s) ∝ 2^{-ell(s)}`.
3. Threaded `grid_codelengths` into shared-mode prior construction in `differentiable_mdl.py`.
4. Added explicit shared logging for `code_cross_entropy_bits` and kept `kl_pi_phi_bits` as an auxiliary diagnostic.
5. Updated shared-MDL help text, metrics reference docs, and active status notes.
6. Replaced shared-MDL tests that assumed the old `1 / (|s| + 1)` heuristic with codelength-based tests and CE/KL identity checks.

## Verification

- `python -m py_compile differentiable_mdl.py src/mdl/shared_weights.py tests/test_mdl.py`
- `./venv/bin/python -m pytest tests/test_mdl.py -q` -> `33 passed`
- shared-mode smoke run completed successfully and printed the corrected shared metrics.

## Notes

The implementation is now aligned with the revised proposal mathematically, but no serious shared-mode retuning has been done yet under the corrected objective.
