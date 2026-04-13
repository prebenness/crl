#!/bin/bash
# JEPA-OT sweep part 2 — 60 minute budget
# Focus: ema_tau crosses at best lambda, probe lower lambda, LR for stability
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

echo "JEPA-OT sweep part 2 started at $(date)"
echo "Deadline: $(date -d @${DEADLINE})"
echo ""

# --- EMA crosses at lambda=0.1 (sweep 1 winner) ---
run_experiment "lam=0.1,tau=0.99"   jepa_ot.lambda_ot=0.1 jepa_ot.ema_tau=0.99   || true
run_experiment "lam=0.1,tau=0.999"  jepa_ot.lambda_ot=0.1 jepa_ot.ema_tau=0.999  || true

# --- Probe lower lambda ---
run_experiment "lam=0.05"           jepa_ot.lambda_ot=0.05                        || true

# --- Lower LR to address instability spikes ---
run_experiment "lam=0.1,lr=5e-4"    jepa_ot.lambda_ot=0.1 training.lr=5e-4       || true

# --- Cross: best lambda × best tau from above ---
run_experiment "lam=0.05,tau=0.99"  jepa_ot.lambda_ot=0.05 jepa_ot.ema_tau=0.99  || true
run_experiment "lam=0.05,tau=0.999" jepa_ot.lambda_ot=0.05 jepa_ot.ema_tau=0.999 || true

# --- EMA crosses at lambda=0.3 (second best) ---
run_experiment "lam=0.3,tau=0.99"   jepa_ot.lambda_ot=0.3 jepa_ot.ema_tau=0.99   || true
run_experiment "lam=0.3,tau=0.999"  jepa_ot.lambda_ot=0.3 jepa_ot.ema_tau=0.999  || true

TOTAL=$(($(date +%s) - START))
echo ""
echo "########## SWEEP 2 DONE: ${RUN_COUNT} runs in ${TOTAL}s ##########"
