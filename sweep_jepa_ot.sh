#!/bin/bash
# JEPA-OT hyperparameter sweep — 45 minute budget
# Priority: lambda_ot sweep first, then ema_tau crosses
set -e

START=$(date +%s)
DEADLINE=$((START + 45*60))
SCRIPT="python jepa_ot.py config/colored_mnist/jepa_ot.yaml"
RUN_COUNT=0

run_experiment() {
    local label="$1"
    shift
    local overrides="$@"

    local now=$(date +%s)
    local remaining=$((DEADLINE - now))

    if [ $remaining -lt 180 ]; then
        echo ""
        echo "########## TIME BUDGET EXHAUSTED: ${remaining}s left, need 180s minimum ##########"
        return 1
    fi

    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo "========== RUN ${RUN_COUNT}: ${label} (${remaining}s remaining) =========="
    echo "Config: ${overrides}"
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
    echo "-----------------"

    local timeout_secs=$((remaining - 120))
    local run_start=$(date +%s)

    timeout ${timeout_secs}s $SCRIPT $overrides
    local exit_code=$?

    local run_end=$(date +%s)
    local run_duration=$((run_end - run_start))

    if [ $exit_code -eq 124 ]; then
        echo ">>> Run timed out after ${run_duration}s"
    elif [ $exit_code -ne 0 ]; then
        echo ">>> Run failed (exit code ${exit_code}) after ${run_duration}s"
    else
        echo ">>> Run completed in ${run_duration}s"
    fi

    return 0
}

echo "JEPA-OT sweep started at $(date)"
echo "Deadline: $(date -d @${DEADLINE})"
echo ""

# --- Lambda sweep (ema_tau=0.996 default) ---
run_experiment "lambda=0.1"  jepa_ot.lambda_ot=0.1  || true
run_experiment "lambda=0.5"  jepa_ot.lambda_ot=0.5  || true
run_experiment "lambda=0.3"  jepa_ot.lambda_ot=0.3  || true
run_experiment "lambda=2.0"  jepa_ot.lambda_ot=2.0  || true

# --- EMA crosses at lambda=0.1 (expected best) ---
run_experiment "lambda=0.1,tau=0.99"   jepa_ot.lambda_ot=0.1 jepa_ot.ema_tau=0.99   || true
run_experiment "lambda=0.1,tau=0.999"  jepa_ot.lambda_ot=0.1 jepa_ot.ema_tau=0.999  || true

# --- If still time: EMA crosses at lambda=0.5 ---
run_experiment "lambda=0.5,tau=0.99"   jepa_ot.lambda_ot=0.5 jepa_ot.ema_tau=0.99   || true
run_experiment "lambda=0.5,tau=0.999"  jepa_ot.lambda_ot=0.5 jepa_ot.ema_tau=0.999  || true

TOTAL=$(($(date +%s) - START))
echo ""
echo "########## SWEEP DONE: ${RUN_COUNT} runs in ${TOTAL}s ##########"
