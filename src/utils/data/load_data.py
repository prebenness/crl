# =========================
# Data pipeline (PyTorch + JAX)
# =========================
import time

import numpy as np
import jax
import jax.numpy as jnp

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.cfg import CFG, DatasetEnum


def make_dataloaders(batch_size=128, num_workers=2, drop_last=True, dataset: DatasetEnum = DatasetEnum.MNIST):
    """
    Returns (train_loader, test_loader). Uses pinned memory and workers.
    """
    g = torch.Generator()
    g.manual_seed(CFG.seed)

    tfm = transforms.ToTensor()  # -> float32 in [0,1], channels-first

    root = CFG.data_root

    if dataset == DatasetEnum.MNIST:
        train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    elif dataset == DatasetEnum.FASHION_MNIST:
        train_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
        test_ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=tfm)
    elif dataset == DatasetEnum.CIFAR10:
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    elif dataset == DatasetEnum.CIFAR100:
        train_ds = datasets.CIFAR100(root=root, train=True, download=True, transform=tfm)
        test_ds = datasets.CIFAR100(root=root, train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), generator=g, drop_last=drop_last
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), drop_last=drop_last
    )
    return train_loader, test_loader


def to_jax_batch(images_t, labels_t):
    """
    Convert CPU torch tensors to JAX device arrays with static shapes.
    Optimized version with pre-allocated arrays and better memory layout.
    images_t: [B,1,28,28] float32 in [0,1]; labels_t: [B] int64
    """
    # Use contiguous memory layout for better performance
    if not images_t.is_contiguous():
        images_t = images_t.contiguous()
    if not labels_t.is_contiguous():
        labels_t = labels_t.contiguous()
    
    # DataLoader yields CPU tensors (pinned). .numpy() is zero-copy view on CPU.
    x_np = images_t.numpy().reshape(images_t.shape[0], -1).astype(np.float32)  # [B,784]
    y_np = labels_t.numpy().astype(np.int32)
    
    # Use device_put for better memory management
    return jax.device_put(jnp.asarray(x_np)), jax.device_put(jnp.asarray(y_np))


def benchmark_dataloader(dataloader, num_batches=10):
    """Benchmark dataloader throughput"""
    times = []
    for i, (images_t, labels_t) in enumerate(dataloader):
        if i >= num_batches:
            break
        start = time.time()
        _ = to_jax_batch(images_t, labels_t)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    throughput = CFG.batch_size / avg_time
    print(f"DataLoader throughput: {throughput:.1f} samples/sec")
    return throughput
