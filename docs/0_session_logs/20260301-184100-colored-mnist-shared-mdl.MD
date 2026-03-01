# ColoredMNIST shared MDL implementation

## Scope
Implemented shared-weight MDL support for the ColoredMNIST ULA code paths, aligning it with the corrected ANBN shared-MDL objective and the updated multi-phase MDL schedule.

## Code changes
- Added `mdl.shared_lambda2` and `mdl.shared_epsilon` to config.
- Added `create_state_mdl_shared(...)` so MDL classifiers can train with both parameter logits and shared `phi_logits`.
- Added shared MDL training/eval steps:
  - `make_train_step_mdl_shared(...)`
  - `make_train_step_mdl_shared_pair(...)`
  - `make_eval_step_mdl_shared(...)`
- The shared objective now uses:
  - inner code term: `CE(pi, phi)`
  - dictionary term: `KL(phi || P_base)`
  - `P_base(s) ∝ 2^{-ell(s)}` derived from the rational code lengths
- Added new model modes in `colored_mnist.py`:
  - `mdl_shared`
  - `mdl_shared_pair`
- Plotting now treats any `*pair` mode as pair-mode for result plots.

## Validation
- `python -m py_compile src/config.py src/training/train_state.py src/training/steps.py colored_mnist.py tests/test_mdl_inner_parity.py`
- `./venv/bin/python -m pytest tests/test_mdl_inner_parity.py -q`
- `./venv/bin/python -m pytest tests/test_training_runners.py -q`

## Smoke runs
- `mdl_shared` single completed successfully on `cmnist_ula` with a 2-epoch fast smoke config.
- `mdl_shared_pair` completed successfully on `cmnist_ula` with a 2-epoch fast smoke config on GPU.
- A full-batch pair smoke (`batch_size=60000`) exceeded GPU memory (~13.4 GiB requested) and was reduced to `8192`, which ran successfully.
