#!/bin/bash
# Baseline experiments: standard LSTM training with CE + optional regularization.
# Reproduces Lan et al. (2024) Section 4.2 baseline comparison.
#
# Configurations:
#   - No regularization (CE only)
#   - L1 with lambda in {0.1, 0.5, 1.0}
#   - L2 with lambda in {0.1, 0.5, 1.0}
#   - Early stopping (patience 2 epochs)
#   - Dropout (0.2)
#
# Each configuration is run across 5 seeds for statistical robustness.
# Total: ~35 runs (some may early-stop quickly)
#
# Usage:
#   bash run_baselines.sh

cd "$(dirname "$0")"

LOG=logs
mkdir -p "$LOG"

PY=python3.12
S=baseline_experiments.py

# Matching Lan et al.: Adam lr=0.001, 20000 epochs, train_size=1000
COMMON="--epochs 20000 --lr 0.001 --num_train 1000 --batch_size 0 \
  --eval_every 100 --log_every 500"

SEEDS="100 200 300 400 500"

run() {
  local name=$1; shift
  echo ""
  echo ">>> Baseline: ${name}"
  $PY $S "$@" $COMMON 2>&1 | tee "${LOG}/baseline_${name}.log"
  echo ">>> Done: ${name}"
}

echo "============================================"
echo "Baseline experiments started at $(date)"
echo "============================================"

# 1. No regularization, no early stopping
for SEED in $SEEDS; do
  run "noreg_s${SEED}" --reg none --seed $SEED
done

# 2. No regularization + early stopping (patience 2 epochs, like Lan et al. best)
for SEED in $SEEDS; do
  run "noreg_es2_s${SEED}" --reg none --early_stop 200 --seed $SEED
done

# 3. L1 regularization
for LAMBDA in 0.1 0.5 1.0; do
  for SEED in $SEEDS; do
    run "l1_lam${LAMBDA}_s${SEED}" --reg l1 --reg_lambda $LAMBDA --seed $SEED
  done
done

# 4. L2 regularization
for LAMBDA in 0.1 0.5 1.0; do
  for SEED in $SEEDS; do
    run "l2_lam${LAMBDA}_s${SEED}" --reg l2 --reg_lambda $LAMBDA --seed $SEED
  done
done

# 5. No reg + dropout (best practical setup from Lan et al.)
for SEED in $SEEDS; do
  run "noreg_do0.2_es2_s${SEED}" --reg none --dropout 0.2 --early_stop 200 --seed $SEED
done

echo ""
echo "============================================"
echo "Baseline experiments complete at $(date)"
echo "============================================"
echo ""
echo "Run 'python3.12 analyze_results.py --tag baseline' to see summary."
