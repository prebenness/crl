#!/bin/bash
# VICReg-OT sweep — 60 minute budget
# Focus: lambda_inv crosses with lambda_var, probe lower lambda_inv, LR for stability
set -e

START=$(date +%s)
DEADLINE=$((START + 60*60))
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

echo "VICReg-OT sweep started at $(date)"
echo "Deadline: $(date -d @${DEADLINE})"
echo ""

# --- Lambda_inv sweep ---
run_experiment "inv=0.1"             jepa_ot.lambda_inv=0.1                         || true
run_experiment "inv=0.3"             jepa_ot.lambda_inv=0.3                         || true
run_experiment "inv=0.5"             jepa_ot.lambda_inv=0.5                         || true
run_experiment "inv=0.05"            jepa_ot.lambda_inv=0.05                        || true

# --- Lambda_var sweep at best lambda_inv ---
run_experiment "inv=0.1,var=10"      jepa_ot.lambda_inv=0.1 jepa_ot.lambda_var=10.0 || true
run_experiment "inv=0.1,var=50"      jepa_ot.lambda_inv=0.1 jepa_ot.lambda_var=50.0 || true

# --- Lambda_cov sweep ---
run_experiment "inv=0.1,cov=0.1"     jepa_ot.lambda_inv=0.1 jepa_ot.lambda_cov=0.1  || true
run_experiment "inv=0.1,cov=5.0"     jepa_ot.lambda_inv=0.1 jepa_ot.lambda_cov=5.0  || true

# --- Lower LR for stability ---
run_experiment "inv=0.1,lr=5e-4"     jepa_ot.lambda_inv=0.1 training.lr=5e-4        || true

TOTAL=$(($(date +%s) - START))
echo ""
echo "########## SWEEP DONE: ${RUN_COUNT} runs in ${TOTAL}s ##########"
