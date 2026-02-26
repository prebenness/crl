"""Tests for ColoredMNIST sweep summary plotting."""

import math

from src.utils.plotting import colored_mnist_plots as plots


class _DummyRun:
    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)


def test_wandb_summary_plot_handles_cmnist_ula_dataset(monkeypatch):
    """Sweep plotting should work for current cmnist_ula runs."""
    captured = {}

    def fake_line_series(**kwargs):
        captured.update(kwargs)
        return {"chart": "ok"}

    monkeypatch.setattr(plots.wandb.plot, "line_series", fake_line_series)

    run = _DummyRun()
    all_data = [
        {
            "dataset": "cmnist_ula",
            "lambda": 1.0e-3,
            "train_acc": 0.99,
            "test_acc": 0.86,
        },
        {
            "dataset": "cmnist_ula",
            "lambda": 1.0e-1,
            "train_acc": 1.00,
            "test_acc": 0.89,
        },
    ]

    plots.wandb_summary_plot(all_data, run)

    assert run.logged
    assert "final_accuracy_plot" in run.logged[0]
    assert "cmnist ula sweep - test" in captured["keys"]

    idx = captured["keys"].index("cmnist ula sweep - test")
    series = captured["ys"][idx]
    assert not all(isinstance(v, float) and math.isnan(v) for v in series)


def test_wandb_summary_plot_keeps_baseline_refs(monkeypatch):
    """Baseline lines are included when lambda=0 rows are present."""
    captured = {}

    def fake_line_series(**kwargs):
        captured.update(kwargs)
        return {"chart": "ok"}

    monkeypatch.setattr(plots.wandb.plot, "line_series", fake_line_series)

    run = _DummyRun()
    all_data = [
        {
            "dataset": "colored_mnist",
            "lambda": 0.0,
            "train_acc": 0.95,
            "test_acc": 0.10,
        },
        {
            "dataset": "colored_mnist",
            "lambda": 1.0e-2,
            "train_acc": 0.90,
            "test_acc": 0.12,
        },
        {
            "dataset": "mnist",
            "lambda": 0.0,
            "train_acc": 0.99,
            "test_acc": 0.98,
        },
    ]

    plots.wandb_summary_plot(all_data, run)

    assert run.logged
    assert "colored mnist baseline (lambda=0) - test" in captured["keys"]
    assert "mnist baseline (lambda=0) - test" in captured["keys"]
