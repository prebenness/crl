# ColoredMNIST warmup-only MDL exploration

## Goal
Check whether MDL (basic/shared) can produce a stable color-only inner model on `cmnist_ula`, and whether an HSIC-regularized outer model can then learn a de-biased classifier from the same raw inputs.

## Key setup change
The critical change was to keep MDL training in the soft phase only:
- `epochs == warmup_epochs`
- no bridge/ST phase is ever entered
- `tau` stays fixed at `2.0`

This avoids the collapse observed in the earlier 40-epoch runs that entered deterministic bridge and ST.

## 1. Single-model inner MDL (color-only target)
### Basic `mdl`
Run: `results/colored_mnist/20260301_223342_cmnist_ula_mdl_warmuponly_basic_cmnist_ula_mdl`

Sweep:
- lambdas: `1e-4`, `5.62e-4`, `3.16e-3`, `1.78e-2`, `1e-1`
- 40 epochs, batch `8192`, `n_max=m_max=10`, `n_samples=3`

Result:
- all 5 lambdas converged to the same endpoint
- final `train_acc ~= 0.912`
- final `test_acc ~= 0.088`

Interpretation:
- this is exactly the desired near-pure color predictor regime
- basic MDL is stable across roughly 3 orders of magnitude in lambda in warmup-only mode

### Shared `mdl_shared`
Run: `results/colored_mnist/20260301_223621_cmnist_ula_mdl_warmuponly_shared_cmnist_ula_mdl_shared`

Same sweep, with:
- `shared_lambda2=20`
- `shared_epsilon=1e-6`

Result:
- same outcome for all 5 lambdas
- final `train_acc ~= 0.912`
- final `test_acc ~= 0.088`

Interpretation:
- corrected shared MDL is also stable across the same 3-order lambda band
- for the inner color-only objective, basic and shared are effectively tied

## 2. Pair model (outer de-biasing with HSIC)
All pair runs below used:
- `lambda = 1e-2`
- 40 epochs
- warmup-only (`warmup_epochs=40`)
- batch `8192`
- `n_max=m_max=10`, `n_samples=3`

### Basic `mdl_pair`
#### `hsic = 0.0`
Run: `results/colored_mnist/20260301_223932_cmnist_ula_mdlpair_warmuponly_hsic0_cmnist_ula_mdl_pair`
- final outer `test_acc = 0.7572`
- best observed outer `test_acc ~= 0.7827`

#### `hsic = 0.5`
Run: `results/colored_mnist/20260301_224116_cmnist_ula_mdlpair_warmuponly_hsic05_cmnist_ula_mdl_pair`
- final outer `test_acc = 0.7864`
- best observed outer `test_acc ~= 0.7996`

#### `hsic = 0.75`
Run: `results/colored_mnist/20260301_225424_cmnist_ula_mdlpair_warmuponly_hsic075_cmnist_ula_mdl_pair`
- final outer `test_acc = 0.7893`
- best observed outer `test_acc ~= 0.8002`

#### `hsic = 0.9`
Run: `results/colored_mnist/20260301_225618_cmnist_ula_mdlpair_warmuponly_hsic09_cmnist_ula_mdl_pair`
- final outer `test_acc = 0.7931`
- best observed outer `test_acc ~= 0.7955`

#### `hsic = 1.0`
Run: `results/colored_mnist/20260301_224258_cmnist_ula_mdlpair_warmuponly_hsic1_cmnist_ula_mdl_pair`
- final outer `test_acc = 0.1201`
- outer model collapses to near chance

Interpretation:
- adding HSIC clearly helps relative to `hsic=0`
- there is a sharp instability threshold between `0.9` and `1.0`
- useful HSIC band appears broad up to `0.9`, then collapses abruptly

### Shared `mdl_shared_pair`
#### `hsic = 0.5`
Run: `results/colored_mnist/20260301_224507_cmnist_ula_mdlsharedpair_warmuponly_hsic05_cmnist_ula_mdl_shared_pair`
- final outer `test_acc = 0.7880`
- best observed outer `test_acc ~= 0.7999`

Interpretation:
- essentially tied with basic `mdl_pair`
- slightly better final value than the matched basic run, but not meaningfully different yet

## Main conclusions
1. The MDL inner model objective is viable for learning a stable color-only predictor on `cmnist_ula`, but only when we do not force it through discretization.
2. In warmup-only mode, both basic and shared MDL are stable across roughly 3 orders of magnitude in lambda for that inner objective.
3. With that stable inner model present, the outer model can reach roughly `0.79` test accuracy under moderate HSIC.
4. HSIC strength has a sharp cliff:
   - workable through at least `0.9`
   - collapse at `1.0`
5. Current best outer result from this round is still far below the `~0.96` target from the comparator papers.

## Practical next steps
1. Add best-checkpoint saving / early stopping for `mdl_pair` and `mdl_shared_pair`, because the best epoch is often a few points better than the final epoch.
2. Run a proper pair lambda sweep in warmup-only mode once GPU init is stable again; current evidence strongly suggests the inner lambda should also be broad in pair mode.
3. Try moderate longer warmup-only runs (e.g. 60 epochs) when GPU is behaving; longer CPU fallbacks were too slow to be useful.
4. Explore outer-model hyperparameters next (learning rate, weight decay, possibly larger `outer_rep_dim`) since the inner side is no longer the main blocker.

## Environment note
CUDA/JAX initialization was flaky during this session:
- some direct interactive runs got `CudaDevice(id=0)` and completed quickly
- some redirected or forced-cuda runs fell back to CPU or failed backend initialization entirely
