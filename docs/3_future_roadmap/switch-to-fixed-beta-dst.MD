# TODO: Switch to Fixed-β DST Training

## Motivation

The current training procedure uses β-annealing (τ from τ_start down to τ_end) combined with a sequence of different gradient estimators (soft → bridge → GST). This means the optimizer is tracking a continuously changing objective landscape. The strongest empirical results appear early in training, before the annealing has had any meaningful effect, which suggests that the annealing mechanism is not contributing what the theory predicts.

The root cause is a mismatch between the theoretical construction and the optimization approach. The paper defines a family of smooth problems J_β parameterized by β. Theorem 1 connects global minimizers across β values. But the implementation tries to solve the whole family simultaneously by sweeping β during training—this couples the annealing schedule to the optimization dynamics in a way the theory doesn't address. β-annealing asks the optimizer to track a moving target, and the empirical evidence shows it fails to do so reliably.

A cleaner approach: solve ONE well-defined optimization problem. Pick a fixed β, optimize J_β to convergence using deterministic straight-through (DST), then read off the discrete solution by argmax.

## What DST gives us

The deterministic ST estimator (already implemented in `src/mdl/lstm.py:126–132`) uses:
- **Forward pass**: hard argmax weights (evaluates the actual discrete network)
- **Backward pass**: soft gradients via `softmax(α/τ)` (smooth gradient proxy)

The straight-through trick: `y_st = y_hard - stop_gradient(y_soft) + y_soft`. The forward value is discrete (y_hard), but gradients flow through y_soft.

Why this is the natural estimator for our problem:
1. The forward pass evaluates L_D at a real discrete point θ = argmax(π). This means the training loss and the validation metric (deterministic accuracy) are always measuring the same thing.
2. Zero forward-pass variance. No Gumbel noise, no stochastic catastrophes.
3. The τ in the backward softmax(α/τ) acts as a fixed smoothing radius controlling how many grid points contribute gradient signal—not as a moving annealing schedule.
4. The coding and entropy terms are computed exactly under π = softmax(α) (analytic, no estimation needed).

## What "fixed β" means concretely

The current objective (from `src/mdl/training.py:198–261`):

```
objective = data_nll_bits + (1/N) * λ * complexity_expected_bits − (1/N) * τ * entropy_weights_bits
```

At fixed β (fixed τ = 1/β), this is a static function of α. The optimizer descends a fixed landscape. Adam's adaptive learning rate handles per-parameter signal-to-noise naturally. There are no phase transitions, no optimizer resets, no schedule coupling.

The τ becomes a hyperparameter to tune (like learning rate or λ), not a schedule to anneal. Its role: larger τ means the backward softmax is more diffuse, so each gradient step is influenced by more grid points (broader search). Smaller τ means gradients are more local to the current argmax (finer refinement). A reasonable starting point: τ = 1.0 (same as the current τ_start).

## Scope of changes

### 1. New training mode in `differentiable_mdl.py`

Add a `--training_mode dst_fixed` option (or similar) alongside the existing phase-based approach. The new mode:

- Creates a single `make_train_step(deterministic_st=True)` instance
- Sets `state.tau = fixed_tau` once at initialization
- Runs a simple loop: `for epoch in range(epochs): state, loss, aux = train_step(...)` + periodic eval
- No phase transitions, no optimizer resets, no tau annealing
- Everything else (eval, checkpointing, logging, analysis) stays the same

The existing phase-based code does not need to be removed—the new mode can coexist as an alternative training path.

### 2. New config file

Create `config/anbn_mdl/dst_fixed.yaml`:

```yaml
mode: basic
training_mode: dst_fixed
n_max: 10
m_max: 10
hidden_size: 3
num_train: 1000
p: 0.3
seed: 0
data_seed: 0

epochs: 12000
lr: 0.03
mdl_lambda: 0.144
tau_fixed: 1.0          # backward-pass smoothing radius (no annealing)
batch_size: 0
init_cl_scale: 1.0

test_max_n: 1500
eval_every: 100
log_every: 200
analyze: true
analyze_max_n: 100000
```

The hyperparameter values above are inherited from the best existing results; `tau_fixed` is a new parameter that replaces `tau_start`/`tau_end`. The key difference: `tau_fixed` doesn't change during training.

### 3. Validation and comparison

Run a sweep of `tau_fixed ∈ {0.3, 0.5, 1.0, 2.0}` and `mdl_lambda ∈ {0.05, 0.144, 1.0}` on the ANBN task. Compare against the best existing results (701 bits at `gen_n = 1,000,000` from the phase-based approach). Metrics:
- `total_mdl_bits` at the best checkpoint
- `gen_n` (perfect generalization length)
- Training stability (does the best checkpoint appear early and then degrade, or is it monotonically improving?)
- Sensitivity to seed (run 3–5 seeds per config)

The most important signal: **does the best checkpoint keep improving throughout training, or does it peak and then degrade?** With fixed-β DST, the landscape doesn't change, so we expect monotonic improvement (modulo Adam noise). If performance still degrades, that's a real local-minimum problem, not a schedule artefact.

### 4. What to do if fixed-β DST gets stuck

The MDL loss surface is known to be rugged (Lan et al. 2024: "hostile to gradient methods"). If fixed-β DST gets stuck at a local minimum that is worse than the best phase-based result, the following mitigations are available (in order of simplicity):

**5a. Multiple restarts.** Run the same config with different seeds. K=63 parameters is small enough that running 10–50 independent seeds is cheap. Take the best result across seeds.

**5b. Population-based training.** Maintain a population of N parallel runs with different initial conditions. Periodically replace the worst performers with perturbed copies of the best. This is a lightweight version of the GA approach used in Lan et al. (2024) and Abudy et al. (2025), but starting from gradient-optimized logits rather than raw weight values.

**5c. Periodic Gumbel perturbation.** Every K epochs, run one epoch of GST (with Gumbel noise) to jolt the optimizer out of a basin. Then resume DST. This adds controlled stochasticity without the complexity of continuous annealing.

**5d. Two-level scheme.** Run gradient-based DST to convergence, extract the argmax weights, then do a small local discrete search (exhaustive over 1–2 weight changes) to check if a nearby discrete solution is better. This hybridises gradient-based and search-based approaches.

## Relationship to the paper

Fixed-β DST is fully consistent with the paper's theoretical framework. The paper defines J_β for each β and proves convergence of global minimizers as β → ∞ (Theorem 1). Optimizing J_β at a single β is the natural computational counterpart—the annealing was always an implementation heuristic, not part of the theory. The paper's discussion (Section 6) already acknowledges that "practical optimisation may still converge to poor local minima, particularly as β increases." Fixed-β DST makes this risk transparent rather than hiding it behind a schedule.

If anything, the paper's theoretical claims are BETTER served by fixed-β DST, because the convergence theorem concerns global minimizers of J_β for a fixed β, not trajectories through a changing landscape. The experimental results can then cleanly characterize: "at this β, our optimizer found this solution" rather than "at some point during our annealing schedule, a checkpoint happened to be good."

## Out of scope for this task

- Reformulating the objective itself (KL-based, CRA penalty, etc.)—these are separate investigations
- Shared-weight mode—adapt DST training there separately after validating on basic mode
- ColoredMNIST—different architecture, different considerations; port after ANBN validation
