#!/bin/bash
cd "$(dirname "$0")"

LOG=logs
PY=/usr/bin/python3.12
S=differentiable_mdl.py
COMMON="--batch_size 0 --eval_every 500 --log_every 200"

run() {
  local name=$1; shift
  echo ""
  echo ">>> Experiment: ${name}"
  $PY $S "$@" $COMMON 2>&1 | tee ${LOG}/${name}.log
  echo ">>> Done: ${name}"
}

echo "Starting remaining experiments at $(date)"

# Long training
run "long_20k" --mode basic --epochs 20000 --warmup_epochs 2000 \
  --lr 0.05 --n_samples 16 --seed 42 \
  --tau_start 1.0 --tau_end 0.005

# Larger grids
run "grid_15x15" --mode basic --epochs 10000 --warmup_epochs 1000 \
  --n_max 15 --m_max 15 \
  --lr 0.05 --n_samples 16 --seed 42 \
  --tau_start 1.0 --tau_end 0.01

run "grid_20x20" --mode basic --epochs 10000 --warmup_epochs 1000 \
  --n_max 20 --m_max 20 \
  --lr 0.05 --n_samples 16 --seed 42 \
  --tau_start 1.0 --tau_end 0.01

# Remaining lambda
run "lambda_3.0" --mode basic --epochs 10000 --warmup_epochs 1000 \
  --lr 0.05 --n_samples 16 --seed 42 \
  --mdl_lambda 3.0 \
  --tau_start 1.0 --tau_end 0.01

# LR sweep
run "lr_0.01" --mode basic --epochs 10000 --warmup_epochs 1000 \
  --lr 0.01 --n_samples 16 --seed 42 \
  --tau_start 1.0 --tau_end 0.01

run "lr_0.1" --mode basic --epochs 10000 --warmup_epochs 1000 \
  --lr 0.1 --n_samples 16 --seed 42 \
  --tau_start 1.0 --tau_end 0.01

# Shared weight mode
run "shared_default" --mode shared --epochs 10000 --warmup_epochs 1000 \
  --lr 0.05 --n_samples 16 --seed 42 \
  --lambda1 1.0 --lambda2 1.0 \
  --tau_start 1.0 --tau_end 0.01

echo ""
echo "All remaining experiments complete at $(date)"
