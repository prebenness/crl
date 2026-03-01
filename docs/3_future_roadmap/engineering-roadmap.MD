# Engineering Roadmap

## Training / Runtime

1. `lax.fori_loop` epoch fusion for full-batch ANBN training
- Reduce Python overhead in long ANBN runs.
- Keep chunk boundaries aligned with logging and evaluation.
- Revisit only where GPU benefits justify the added XLA complexity.

2. Vectorize `_compute_discrete_hyp_bits`
- Replace Python loops over discrete weights with lookup-based indexing.
- Low risk, modest runtime improvement.

## Evaluation Correctness

3. Use full fixed test sets during evaluation
- Stop dropping remainder batches.
- Keep test iteration deterministic and complete.

4. Improve train tail-batch handling
- Stop dropping remainder samples in training.
- Either mask the tail batch or rotate offsets across epochs.

## Dataset / Config Consistency

5. Clarify `p_test` semantics for `cmnist_ula`
- Either ignore it explicitly with a warning or wire it into alternate dataset builders.
- Keep docs and code aligned.

6. Enforce or truly support `rep_dim` width configuration
- `ULAMLPClassifier` should either require `rep_dim == 100` or use the config value consistently.

## Reporting / Metrics

7. Improve shared-MDL reporting semantics
- In shared mode, report both discrete `|H|` and shared-objective compression terms clearly.
- Avoid implying that a single `|H|` fully describes the shared objective.
