# CRL Experiments

This repository contains two main experiment tracks:

- `differentiable_mdl.py`: `a^n b^n` language-learning experiments with differentiable MDL objectives
- `colored_mnist.py`: ColoredMNIST sweeps with VIB/HSIC and MDL-based models

Standalone debiasing methods:

- `jepa_ot.py`: JEPA-OT (Action-Conditioned JEPA with Latent Counterfactuals via Sinkhorn Alignment)
- `s3e.py`: S3E (Spectral Spurious Subspace Elimination)
- `fg_ccdb_replication.py`: FG-CCDB replication (Zhao et al. 2025, arXiv:2505.06831v1)

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

Run JEPA-OT (cMNIST):

```bash
python jepa_ot.py config/colored_mnist/jepa_ot.yaml
```

Run JEPA-OT (cCIFAR-10):

```bash
python jepa_ot.py config/ccifar10/jepa_ot.yaml
```

Override any param via CLI: `jepa_ot.lambda_ot=0.5 jepa_ot.ema_tau=0.999 training.lr=5e-4`
