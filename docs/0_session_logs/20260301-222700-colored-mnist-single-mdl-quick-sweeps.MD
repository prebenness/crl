# ColoredMNIST quick single-model MDL sweeps

## Scope
Ran quick 40-epoch `cmnist_ula` sweeps for single-model ULA MDL under the updated multi-phase schedule.

## Config
Shared settings for both sweeps:
- dataset: `cmnist_ula`
- epochs: `40`
- batch_size: `8192`
- lr: `1e-3`
- MDL grid: `n_max=10`, `m_max=10`
- `tau_start=2.0`, `tau_end=0.1`
- `warmup_epochs=20`
- `n_samples=3`
- lambdas: `1e-3`, `3.16e-3`, `1e-2`, `3.16e-2`, `1e-1`

Shared-only fixed hyperparameters:
- `shared_lambda2=20`
- `shared_epsilon=1e-6`

## Results
Both `mdl` and `mdl_shared` single-model sweeps collapsed after the bridge/ST transition.

Final metrics for all 5 lambdas in both sweeps:
- train accuracy: approximately `0.125`
- test accuracy: approximately `0.088`

This means the new schedule that helped ANBN is not directly working for the single-model ColoredMNIST MDL path under this short 40-epoch setup.

## Interpretation
- Warmup phase still learns the spurious/color shortcut (train accuracy rises during warmup).
- The deterministic bridge and subsequent ST phase wipe out that learned basin in the single-model setting.
- The corrected shared objective does not fix this by itself at this timescale.

## Relevant run directories
- basic: `results/colored_mnist/20260301_221335_cmnist_ula_mdl_quick_basic_cmnist_ula_mdl`
- shared: `results/colored_mnist/20260301_221807_cmnist_ula_mdl_quick_shared_cmnist_ula_mdl_shared`
