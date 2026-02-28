"""Tests for dataset materialisation and dataloader config wiring."""

import numpy as np
import pytest

from src.config import load_config
from src.datasets import datasets as datasets_mod


class _ToyDataset:
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        x = np.full((1, 2, 2), idx, dtype=np.float32)
        return x, np.int32(idx)


def test_load_config_reads_dataloader_section():
    """Colored-MNIST configs should populate the dataloader section."""
    cfg = load_config("config/colored_mnist/mdl_single_sweep.yaml")

    assert cfg.dataloader.num_workers == 0
    assert cfg.dataloader.pin_memory is False
    assert cfg.dataloader.persistent_workers is False
    assert cfg.dataloader.prefetch_factor == 2


def test_dataset_to_jax_arrays_omits_worker_only_kwargs_when_disabled(monkeypatch):
    """Worker-only DataLoader kwargs should stay unset when workers=0."""
    captured = {}

    class _FakeLoader:
        def __init__(self, dataset, **kwargs):
            captured["kwargs"] = kwargs

        def __iter__(self):
            yield np.array(
                [
                    [[[0.0, 0.0], [0.0, 0.0]]],
                    [[[1.0, 1.0], [1.0, 1.0]]],
                ],
                dtype=np.float32,
            ), np.array([0, 1], dtype=np.int32)
            yield np.array(
                [[[[2.0, 2.0], [2.0, 2.0]]]],
                dtype=np.float32,
            ), np.array([2], dtype=np.int32)

    monkeypatch.setattr(datasets_mod, "DataLoader", _FakeLoader)

    x, y = datasets_mod.dataset_to_jax_arrays(
        _ToyDataset(),
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=7,
    )

    assert captured["kwargs"]["batch_size"] == 2
    assert captured["kwargs"]["num_workers"] == 0
    assert captured["kwargs"]["pin_memory"] is True
    assert "persistent_workers" not in captured["kwargs"]
    assert "prefetch_factor" not in captured["kwargs"]
    assert np.asarray(x).shape == (3, 2, 2, 1)
    assert np.asarray(y).tolist() == [0, 1, 2]


def test_dataset_to_jax_arrays_passes_worker_kwargs(monkeypatch):
    """Worker settings should be forwarded when multiprocessing is enabled."""
    captured = {}

    class _FakeLoader:
        def __init__(self, dataset, **kwargs):
            captured["kwargs"] = kwargs

        def __iter__(self):
            yield np.array(
                [[[[0.0, 0.0], [0.0, 0.0]]]],
                dtype=np.float32,
            ), np.array([0], dtype=np.int32)
            yield np.array(
                [[[[1.0, 1.0], [1.0, 1.0]]]],
                dtype=np.float32,
            ), np.array([1], dtype=np.int32)
            yield np.array(
                [[[[2.0, 2.0], [2.0, 2.0]]]],
                dtype=np.float32,
            ), np.array([2], dtype=np.int32)

    monkeypatch.setattr(datasets_mod, "DataLoader", _FakeLoader)

    datasets_mod.dataset_to_jax_arrays(
        _ToyDataset(),
        batch_size=8,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=5,
    )

    assert captured["kwargs"]["batch_size"] == 3
    assert captured["kwargs"]["num_workers"] == 2
    assert captured["kwargs"]["persistent_workers"] is True
    assert captured["kwargs"]["prefetch_factor"] == 5


def test_dataset_to_jax_arrays_rejects_invalid_persistent_workers():
    """persistent_workers=True requires multiprocessing workers."""
    with pytest.raises(ValueError, match="persistent_workers"):
        datasets_mod.dataset_to_jax_arrays(
            _ToyDataset(),
            batch_size=2,
            num_workers=0,
            persistent_workers=True,
        )
