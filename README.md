# CRL Experiments

This repository contains two main experiment tracks:

- `differentiable_mdl.py`: `a^n b^n` language-learning experiments with differentiable MDL objectives
- `colored_mnist.py`: ColoredMNIST sweeps with VIB/HSIC and MDL-based models

## Documentation

Project documentation now lives under `docs/`.

Start here:
- [docs/README.MD](docs/README.MD)

Most useful current files:
- `docs/2_active_discussions/anbn-mdl-current-status.MD`
- `docs/2_active_discussions/shared-mdl-status.MD`
- `docs/2_active_discussions/colored-mnist-current-status.MD`
- `docs/1_closed_topics/mdl-metrics-and-logging.MD`

## Quick Commands

Run ANBN MDL:

```bash
python differentiable_mdl.py config/anbn_mdl/basic_train.yaml
```

Run ColoredMNIST:

```bash
python colored_mnist.py config/colored_mnist/vib_pair_sweep.yaml
```
