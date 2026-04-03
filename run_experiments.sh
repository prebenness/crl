#!/bin/bash
# Sequential experiment sweep — cMNIST oracle pair across bias ratios + cCIFAR10
# Runs one at a time to keep GPU usage low.
set -e

ORACLE_CKPT="results/colored_mnist/20260331_172654_oracle_train_cmnist_ula_oracle_train/runs/lambda_1.0e+00/checkpoints/best.npz"
RESULTS_LOG="experiment_results.txt"

echo "=== Experiment Sweep $(date) ===" | tee "$RESULTS_LOG"

# ─── cMNIST: ERM baselines across bias ratios ───
for BETA in 0.005 0.01 0.02 0.05; do
    P_TRAIN=$(python -c "print(1 - $BETA)")
    echo "" | tee -a "$RESULTS_LOG"
    echo "--- cMNIST ERM β=$BETA (p_train=$P_TRAIN) ---" | tee -a "$RESULTS_LOG"
    python colored_mnist.py config/colored_mnist/erm_baseline.yaml \
        dataset.p_train=$P_TRAIN \
        training.epochs=50 \
        2>&1 | grep -E "NEW BEST|^cmnist|Lambda|^1\." | tee -a "$RESULTS_LOG"
done

# ─── cMNIST: Oracle pair across bias ratios ───
for BETA in 0.005 0.01 0.02 0.05; do
    P_TRAIN=$(python -c "print(1 - $BETA)")
    echo "" | tee -a "$RESULTS_LOG"
    echo "--- cMNIST Oracle Pair β=$BETA (p_train=$P_TRAIN) hsic=0.7 ---" | tee -a "$RESULTS_LOG"
    python colored_mnist.py config/colored_mnist/oracle_pair.yaml \
        dataset.p_train=$P_TRAIN \
        training.epochs=200 \
        hsic.weight=0.7 \
        checkpointing.restart_patience=15 \
        2>&1 | grep -E "NEW BEST|RESTART|^cmnist|Lambda|^1\." | tee -a "$RESULTS_LOG"
done

# ─── cCIFAR10: ERM baseline (ResNet18, β=0.05) ───
echo "" | tee -a "$RESULTS_LOG"
echo "--- cCIFAR10 ERM β=0.05 ResNet18 ---" | tee -a "$RESULTS_LOG"
python train.py config/ccifar10/resnet18_single.yaml \
    training.epochs=50 \
    2>&1 | grep -E "NEW BEST|Epoch 50|^ccifar|Lambda|^1\." | tee -a "$RESULTS_LOG"

# ─── cCIFAR10: ERM at other bias ratios ───
for BETA in 0.01 0.02; do
    P_TRAIN=$(python -c "print(1 - $BETA)")
    echo "" | tee -a "$RESULTS_LOG"
    echo "--- cCIFAR10 ERM β=$BETA ResNet18 ---" | tee -a "$RESULTS_LOG"
    python train.py config/ccifar10/resnet18_single.yaml \
        dataset.p_train=$P_TRAIN \
        training.epochs=50 \
        2>&1 | grep -E "NEW BEST|Epoch 50|^ccifar|Lambda|^1\." | tee -a "$RESULTS_LOG"
done

echo "" | tee -a "$RESULTS_LOG"
echo "=== All experiments complete $(date) ===" | tee -a "$RESULTS_LOG"
