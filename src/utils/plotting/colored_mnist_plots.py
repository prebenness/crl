import numpy as np
import wandb
import pandas as pd


def wandb_summary_plot(all_data, wandb_run):
    """
    Log a summary chart: accuracy vs lambda for
    ColoredMNIST sweep (train/test) + horizontal baseline refs.

    all_data: list of dicts with keys including
              ["dataset", "lambda", "train_acc", "test_acc", ...]
    """
    if not all_data:
        return

    df = pd.DataFrame(all_data)
    df["lambda"] = df["lambda"].map(float)

    # X axis: use all lambdas we observed (includes 0.0 baseline)
    lambdas = sorted(df["lambda"].unique())
    xs = lambdas

    def _baseline_value(ds_code: str, acc_key: str):
        # Baseline runs are identified by lambda == 0.0 in this script
        row = df[(df["dataset"] == ds_code) & (np.isclose(df["lambda"], 0.0))]
        if len(row) == 0:
            return np.nan
        return float(row[acc_key].iloc[0])

    def _sweep_series(ds_code: str, acc_key: str):
        # For the sweep curve, we intentionally skip the baseline point at lambda==0
        series = []
        for lmb in xs:
            if np.isclose(lmb, 0.0):
                series.append(np.nan)  # gap at baseline
                continue
            row = df[(df["dataset"] == ds_code) & (np.isclose(df["lambda"], lmb))]
            if len(row) == 0:
                series.append(np.nan)
            else:
                series.append(float(row[acc_key].iloc[0]))
        return series

    # ---- Build curves ----
    ys = []
    keys = []

    # ColoredMNIST sweep (train/test)
    keys += ["ColoredMNIST sweep – train", "ColoredMNIST sweep – test"]
    ys += [
        _sweep_series("colored_mnist", "train_acc"),
        _sweep_series("colored_mnist", "test_acc"),
    ]

    # ColoredMNIST baselines as horizontal reference lines
    c_tr0 = _baseline_value("colored_mnist", "train_acc")
    c_te0 = _baseline_value("colored_mnist", "test_acc")
    keys += ["ColoredMNIST baseline (λ=0) – train", "ColoredMNIST baseline (λ=0) – test"]
    ys += [
        [c_tr0] * len(xs),
        [c_te0] * len(xs),
    ]

    # MNIST baselines as horizontal reference lines (optional but handy)
    m_tr0 = _baseline_value("mnist", "train_acc")
    m_te0 = _baseline_value("mnist", "test_acc")
    keys += ["MNIST baseline (λ=0) – train", "MNIST baseline (λ=0) – test"]
    ys += [
        [m_tr0] * len(xs),
        [m_te0] * len(xs),
    ]

    chart = wandb.plot.line_series(
        xs=xs,
        ys=ys,
        keys=keys,
        title="Accuracy vs λ (ColoredMNIST sweep + baselines)",
        xname="lambda",
    )
    wandb_run.log({"final_accuracy_plot": chart})