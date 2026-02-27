#!/bin/bash
# Experiment battery for differentiable MDL.
# Expected total runtime: ~60 minutes.
# Each experiment logs to logs/<name>.log.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG="$ROOT_DIR/logs"
mkdir -p "$LOG"
PY="${PY:-python3.12}"
S="$ROOT_DIR/differentiable_mdl.py"
CFG="$ROOT_DIR/config/anbn_mdl/basic_train.yaml"

# Common: full batch, eval every 500, log every 200
COMMON="--batch_size 0 --eval_every 500 --log_every 200"

run() {
  local name=$1; shift
  echo ""
  echo ">>> Experiment: ${name}"
  $PY $S "$CFG" "$@" $COMMON 2>&1 | tee "${LOG}/${name}.log"
  echo ">>> Done: ${name}"
}

echo "============================================"
echo "Starting experiment battery at $(date)"
echo "============================================"

# 1. SEED SWEEP (3 seeds x 10k epochs) — is gen_n=100 robust or lucky?
for SEED in 42 0 123; do
  run "seed_${SEED}_10k" --mode basic --epochs 10000 --warmup_epochs 1000 \
    --lr 0.05 --n_samples 16 --seed $SEED \
    --tau_start 1.0 --tau_end 0.01
done

# 2. VERY LONG TRAINING (20k epochs) — are we still improving at 10k?
run "long_20k" --mode basic --epochs 20000 --warmup_epochs 2000 \
  --lr 0.05 --n_samples 16 --seed 42 \
  --tau_start 1.0 --tau_end 0.005

# 3. LARGER GRID — more expressiveness
run "grid_15x15" --mode basic --epochs 10000 --warmup_epochs 1000 \
  --n_max 15 --m_max 15 \
  --lr 0.05 --n_samples 16 --seed 42 \
  --tau_start 1.0 --tau_end 0.01

run "grid_20x20" --mode basic --epochs 10000 --warmup_epochs 1000 \
  --n_max 20 --m_max 20 \
  --lr 0.05 --n_samples 16 --seed 42 \
  --tau_start 1.0 --tau_end 0.01

# 4. LAMBDA SWEEP — how does MDL penalty affect generalization?
for LAMBDA in 0.1 0.5 3.0; do
  run "lambda_${LAMBDA}" --mode basic --epochs 10000 --warmup_epochs 1000 \
    --lr 0.05 --n_samples 16 --seed 42 \
    --mdl_lambda $LAMBDA \
    --tau_start 1.0 --tau_end 0.01
done

# 5. LEARNING RATE SWEEP
for LR in 0.01 0.1; do
  run "lr_${LR}" --mode basic --epochs 10000 --warmup_epochs 1000 \
    --lr $LR --n_samples 16 --seed 42 \
    --tau_start 1.0 --tau_end 0.01
done

# 6. SHARED WEIGHT MODE — untested
run "shared_default" --mode shared --epochs 10000 --warmup_epochs 1000 \
  --lr 0.05 --n_samples 16 --seed 42 \
  --lambda1 1.0 --lambda2 1.0 \
  --tau_start 1.0 --tau_end 0.01

echo ""
echo "============================================"
echo "All experiments complete at $(date)"
echo "============================================"
