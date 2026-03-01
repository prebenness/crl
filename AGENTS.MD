# AGENTS.md

This file provides operational guidance for tool-driven work in this repository.

## Always Do

1. Start by reading the relevant documents in `docs/`.
2. At the end of every major piece of work, add a timestamped file to `docs/0_session_logs/`.

## Documentation Entry Point

- `docs/README.MD`

## Core Commands

Run tests:

```bash
pytest tests/test_mdl.py -v
pytest tests/test_math.py -v
```

Run ANBN MDL:

```bash
python3.12 differentiable_mdl.py config/anbn_mdl/basic_train.yaml
```

Run ColoredMNIST:

```bash
python colored_mnist.py config/colored_mnist/vib_pair_sweep.yaml
```

Run full experiment battery:

```bash
bash run_experiments.sh
```

## Important Note

Keep root-level docs minimal. Stable reference material, active technical status, session logs, and planning now belong in `docs/`.
