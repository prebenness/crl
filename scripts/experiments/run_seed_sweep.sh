#!/bin/bash
# Seed sweep for differentiable MDL — best configuration across 10 seeds.
# Uses the configuration from the best existing run (gen_n=18540, seed=0).
#
# Usage:
#   bash run_seed_sweep.sh          # run all 10 seeds
#   bash run_seed_sweep.sh 0 4      # run seeds 0..4 only

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG="$ROOT_DIR/logs"
mkdir -p "$LOG"

PY="${PY:-python3.12}"
S="$ROOT_DIR/differentiable_mdl.py"
CFG="$ROOT_DIR/config/anbn_mdl/basic_train.yaml"

# Best configuration from existing runs
COMMON="--mode basic --epochs 10000 --warmup_epochs 1000 \
  --lr 0.05 --mdl_lambda 1.0 --n_max 10 --m_max 10 \
  --n_samples 16 --tau_start 1.0 --tau_end 0.01 \
  --batch_size 0 --eval_every 100 --log_every 200 \
  --analyze --analyze_max_n 100000"

# Seed range (default: 0..9)
SEED_START=${1:-0}
SEED_END=${2:-9}

run() {
  local seed=$1
  local name="sweep_seed_${seed}"
  echo ""
  echo ">>> Seed sweep: seed=${seed}"
  echo ">>> Log: ${LOG}/${name}.log"
  $PY $S "$CFG" $COMMON --seed $seed 2>&1 | tee "${LOG}/${name}.log"
  echo ">>> Done: seed=${seed}"
}

echo "============================================"
echo "MDL seed sweep: seeds ${SEED_START}..${SEED_END}"
echo "Started at $(date)"
echo "============================================"

for SEED in $(seq $SEED_START $SEED_END); do
  run $SEED
done

echo ""
echo "============================================"
echo "Seed sweep complete at $(date)"
echo "============================================"
echo ""
echo "Run 'python3.12 scripts/analysis/summarize_experiment_results.py --tag sweep' to see summary statistics."
