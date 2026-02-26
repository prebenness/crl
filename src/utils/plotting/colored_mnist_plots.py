import numpy as np
import wandb
import pandas as pd


def wandb_summary_plot(all_data, wandb_run):
    """
    Log a summary chart: accuracy vs lambda for any dataset present in
    the sweep, with optional horizontal baseline references (lambda==0).

    all_data: list of dicts with keys including
              ["dataset", "lambda", "train_acc", "test_acc", ...]
    """
    if not all_data:
        return

    df = pd.DataFrame(all_data)
    required = {"dataset", "lambda", "train_acc", "test_acc"}
    if not required.issubset(df.columns):
        return

    df = df[list(required)].copy()
    df["dataset"] = df["dataset"].astype(str)
    df["lambda"] = df["lambda"].map(float)
    df["train_acc"] = df["train_acc"].map(float)
    df["test_acc"] = df["test_acc"].map(float)
    # If duplicate rows exist for the same dataset/lambda, average them.
    df = (
        df.groupby(["dataset", "lambda"], as_index=False)[
            ["train_acc", "test_acc"]
        ]
        .mean()
    )

    # X axis: use all lambdas observed across datasets (includes 0.0 baseline)
    lambdas = sorted(df["lambda"].unique())
    xs = lambdas

    def _baseline_value(ds_name: str, acc_key: str):
        row = df[(df["dataset"] == ds_name) & (np.isclose(df["lambda"], 0.0))]
        if len(row) == 0:
            return np.nan
        return float(row[acc_key].iloc[0])

    def _sweep_series(ds_name: str, acc_key: str):
        # For the sweep curve, skip the baseline point at lambda == 0.
        series = []
        for lmb in xs:
            if np.isclose(lmb, 0.0):
                series.append(np.nan)  # gap at baseline
                continue
            row = df[(df["dataset"] == ds_name) & (np.isclose(df["lambda"], lmb))]
            if len(row) == 0:
                series.append(np.nan)
            else:
                series.append(float(row[acc_key].iloc[0]))
        return series

    def _all_nan(vals):
        return all(np.isnan(v) for v in vals)

    # ---- Build curves dynamically from datasets present ----
    ys = []
    keys = []
    datasets = list(df["dataset"].unique())

    for ds_name in datasets:
        label = ds_name.replace("_", " ")
        sweep_train = _sweep_series(ds_name, "train_acc")
        sweep_test = _sweep_series(ds_name, "test_acc")

        if not _all_nan(sweep_train):
            keys.append(f"{label} sweep - train")
            ys.append(sweep_train)
        if not _all_nan(sweep_test):
            keys.append(f"{label} sweep - test")
            ys.append(sweep_test)

        baseline_train = _baseline_value(ds_name, "train_acc")
        baseline_test = _baseline_value(ds_name, "test_acc")
        if not np.isnan(baseline_train):
            keys.append(f"{label} baseline (lambda=0) - train")
            ys.append([baseline_train] * len(xs))
        if not np.isnan(baseline_test):
            keys.append(f"{label} baseline (lambda=0) - test")
            ys.append([baseline_test] * len(xs))

    if not ys:
        return

    chart = wandb.plot.line_series(
        xs=xs,
        ys=ys,
        keys=keys,
        title="Accuracy vs lambda (sweep + baselines)",
        xname="lambda",
    )
    wandb_run.log({"final_accuracy_plot": chart})
