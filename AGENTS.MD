# AGENTS.md

This file provides operational guidance for tool-driven work in this repository.

## Always Do

1. Start by reading the relevant documents in `docs/`.
2. At the end of every major piece of work, add a timestamped file to `docs/0_session_logs/`.

## Documentation Entry Point

- `docs/README.MD`

## Repository Structure

This repo contains the spurious correlation classification experiments. The MDL formal language learning code lives in its own repo, imported as a git submodule at `MDL/`. The classification path imports a small number of functions from `MDL.src.mdl` (grid construction, shared weights, tau annealing).

## Core Commands

Run tests:

```bash
pytest tests/ -v
```

Run ColoredMNIST:

```bash
python colored_mnist.py config/colored_mnist/vib_pair_sweep.yaml
```

## Important Note

Keep root-level docs minimal. Stable reference material, active technical status, session logs, and planning now belong in `docs/`.
