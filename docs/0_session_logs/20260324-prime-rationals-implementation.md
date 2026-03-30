# 2026-03-24: Prime-Exponent Relaxation Implementation

## What was done

Implemented a new weight parameterization for MDL-style neural network regularization based on continuous prime-exponent relaxation. Each weight is:

```
s_i = tanh(u_i) * exp(clamp(z_i^T log(primes)))
```

with MDL regularizer `R = sum |z_{i,r}| * log(p_r)`.

This is an alternative to the existing Gumbel-Softmax categorical grid approach — fully differentiable, no sampling needed, and the MDL penalty becomes a weighted L1 norm on exponent coordinates.

## Files created

- `prime_rationals.py` — Main implementation (entry point + all new code)
  - Math utilities: `first_primes`, `reconstruct_weight`, `compute_mdl_penalty`, `discretize`, `exponents_to_rational`
  - Flax modules: `PrimeExpLinear`, `PrimeExpMLP` (XOR toy), `PrimeExpLSTM` (ANBN)
  - Loss functions, JIT-compiled train steps, evaluation wrappers
  - Post-training discretization and rational weight printing
  - Config dataclass + YAML loading + CLI entry point
- `config/prime_rationals/xor.yaml` — XOR toy experiment config
- `config/prime_rationals/anbn.yaml` — ANBN experiment config
- `docs/3_future_roadmap/prime_rationals_todo.md` — Deferred extensions (hard sign STE, Xavier/He init, zero gates, layer scale, pos/neg split, layerwise primes, lambda annealing)

## Design decisions

1. JAX/Flax (consistent with existing codebase)
2. Smooth sign via `tanh(u_i)` (hard sign STE deferred)
3. P (number of primes) as config parameter, default 6
4. Simple N(0, 0.01) initialization (Xavier/He deferred)
5. No zero gates, layer scale, or pos/neg split for v1

## Reuse from existing repo

- ANBN data pipeline (`src/mdl/data`)
- Evaluation metrics (`src/mdl/evaluation`): `compute_per_string_nll_bits`, `compute_grammar_weighted_nll_bits`, `compute_train_dh`
- Run management (`src/utils/checkpointing`): `TeeLogger`, `make_experiment_dir`, `checkpoint_path`, etc.
- LSTM architecture mirrors `GumbelSoftmaxLSTM` weight layout exactly (108 params for H=I=O=3)

## Test results

Both XOR and ANBN experiments run end-to-end:
- XOR (5000 epochs, P=6, lambda=0.001): 97% continuous accuracy, MDL penalty decreases steadily
- ANBN (500 epochs quick test, P=6, lambda=100): 95% test accuracy after short training

Note: with current default configs, the regularizer pushes exponents very close to zero, so weights are near ±1. Lambda tuning needed for richer rational structure to emerge.

## Next steps

- Hyperparameter tuning: find lambda values where non-trivial rational structure emerges
- Longer ANBN training runs for comparison against Gumbel-Softmax results
- Extensions from TODO doc as needed
