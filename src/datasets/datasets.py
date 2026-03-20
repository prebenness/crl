import warnings

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

import numpy as np
import jax.numpy as jnp

# ============================================================
# 1) Custom PyTorch datasets
# ============================================================

class ColoredMNIST(Dataset):
    """
    Standard ColoredMNIST:
      - Binary label: y_bin = (digit < 5)
      - Color correlates with y_bin with prob p_corr
      - Returns (x_rgb, y_bin) with x_rgb in CHW.
    """
    def __init__(self, root="./data", train=True, p_corr=0.9, seed=0):
        super().__init__()
        self.rng = np.random.RandomState(seed)

        base = MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        images = base.data.numpy().astype(np.float32) / 255.0  # [N,28,28]
        labels = base.targets.numpy().astype(np.int32)

        y_bin = (labels < 5).astype(np.int32)

        flip = self.rng.rand(len(y_bin)) > p_corr
        colors = np.where(flip, 1 - y_bin, y_bin).astype(np.int32)

        self.images = images
        self.y_bin = y_bin
        self.colors = colors

    def __len__(self):
        return len(self.y_bin)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.y_bin[idx]
        c = self.colors[idx]

        xr = x if c == 0 else np.zeros_like(x)
        xg = x if c == 1 else np.zeros_like(x)
        xb = np.zeros_like(x)

        x_rgb = np.stack([xr, xg, xb], axis=0)  # [3,28,28] CHW
        return x_rgb, y


class IRMColoredMNIST(Dataset):
    """
    IRM-style ColoredMNIST:
      - Binary label (clean): y_clean = (digit < 5)
      - Noisy label: y = y_clean flipped with prob 0.25
      - Color correlates with *noisy* y with prob p_corr
      - Returns (x_rgb, y) with x_rgb in CHW.
    """
    LABEL_FLIP_PROB = 0.25  # P(y != y_clean)

    def __init__(self, root="./data", train=True, p_corr=0.9, seed=0):
        super().__init__()
        self.rng = np.random.RandomState(seed)

        base = MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        # [N, 28, 28], float32 in [0, 1]
        images = base.data.numpy().astype(np.float32) / 255.0
        labels = base.targets.numpy().astype(np.int32)

        # 1. Clean binary label from digit
        #    (you can swap <5 / >=5 if you prefer the other convention)
        y_clean = (labels < 5).astype(np.int32)

        # 2. Add label noise: flip with probability LABEL_FLIP_PROB
        flip_y = self.rng.rand(len(y_clean)) < self.LABEL_FLIP_PROB
        y_noisy = np.where(flip_y, 1 - y_clean, y_clean).astype(np.int32)

        # 3. Color correlated with *noisy* label with probability p_corr
        #    (same semantics as your original: p_corr = P(color == y_noisy))
        flip_c = self.rng.rand(len(y_noisy)) > p_corr
        colors = np.where(flip_c, 1 - y_noisy, y_noisy).astype(np.int32)

        self.images = images            # [N, 28, 28], grayscale
        self.y_clean = y_clean          # optional: clean label (for analysis)
        self.y_bin = y_noisy            # noisy label used for training
        self.colors = colors            # color indicator (0=red, 1=green)

    def __len__(self):
        return len(self.y_bin)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.y_bin[idx]
        c = self.colors[idx]

        xr = x if c == 0 else np.zeros_like(x)
        xg = x if c == 1 else np.zeros_like(x)
        xb = np.zeros_like(x)

        x_rgb = np.stack([xr, xg, xb], axis=0)  # [3, 28, 28] CHW
        return x_rgb, y


class StandardMNISTBin(Dataset):
    """Binary MNIST (digit<5), grayscale CHW."""
    def __init__(self, root="./data", train=True):
        base = MNIST(root=root, train=train, download=True, transform=transforms.ToTensor())
        images = base.data.numpy().astype(np.float32) / 255.0
        labels = base.targets.numpy().astype(np.int32)
        y_bin = (labels < 5).astype(np.int32)

        self.images = images
        self.y_bin = y_bin

    def __len__(self):
        return len(self.y_bin)

    def __getitem__(self, idx):
        x = self.images[idx]
        x = np.expand_dims(x, 0)  # [1,28,28]
        return x, self.y_bin[idx]


class CMNISTuLA(Dataset):
    """
    Option A) cMNIST (uLA / CCDB protocol).

    Matches uLA Appendix D.2:
      - Target: digit y in {0..9}
      - Bias: color z in {0..9}, with a designated paired color per digit
      - Development (train=True): P(z = paired(y)) = 1 - beta, else uniform over remaining 9 colors
        In this class we expose p_corr := 1 - beta (so beta = 1 - p_corr).
      - Test (train=False): unbiased; digit-color pairs random ("random chance 1/10^2"),
        implemented as z ~ Uniform({0..9}) independent of y.

    Signature matches your existing ColoredMNIST:
        __init__(root="./data", train=True, p_corr=0.9, seed=0)

    Returns:
        (x_rgb, y) where x_rgb is float32 CHW in [0,1], y is int32 in {0..9}.

    Notes / assumptions (not specified in uLA/CCDB PDFs):
      - Exact RGB palette is not standardized in the papers; we provide a fixed 10-color palette.
      - We colorize the *foreground digit* (background stays black). If their implementation colors
        the background instead, you can switch the application rule later without changing the
        correlation structure.
      - Train/val split is not encoded here; create val by splitting indices from the biased
        train=True dataset object (so val is also from pdata, as described).
    """

    # 10 distinct RGB colors in [0,1]. You can change these later to match an official implementation.
    COLOR_PALETTE = np.array([
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 1.0],  # magenta
        [0.0, 1.0, 1.0],  # cyan
        [1.0, 0.5, 0.0],  # orange
        [0.5, 0.0, 1.0],  # purple
        [0.6, 0.3, 0.0],  # brown-ish
        [0.2, 0.8, 0.2],  # light green
    ], dtype=np.float32)

    # Paired color per digit. Identity mapping matches the “k paired with a distinct color” description.
    PAIRED_COLOR = np.arange(10, dtype=np.int32)

    def __init__(self, root="./data", train=True, p_corr=0.9, seed=0):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.train_split = bool(train)

        base = MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        images = base.data.numpy().astype(np.float32) / 255.0  # [N,28,28]
        labels = base.targets.numpy().astype(np.int32)         # [N], values 0..9

        y = labels
        paired = self.PAIRED_COLOR[y]  # [N] in 0..9

        if self.train_split:
            # Biased development distribution: z = paired(y) w.p. p_corr, else uniform over other 9 colors.
            aligned = self.rng.rand(len(y)) < float(p_corr)

            # Sample a "shift" in {1..9} uniformly; (paired + shift) mod 10 gives a color != paired,
            # uniformly over the remaining 9 options.
            shift = 1 + self.rng.randint(0, 9, size=len(y)).astype(np.int32)
            other = (paired + shift) % 10

            colors = np.where(aligned, paired, other).astype(np.int32)
        else:
            # Unbiased test: z ~ Uniform({0..9}) independent of y.
            colors = self.rng.randint(0, 10, size=len(y)).astype(np.int32)

        self.images = images
        self.y = y
        self.colors = colors

        # Convenience: group id for (y,z) if you want group-balanced eval later.
        self.group = (self.y.astype(np.int32) * 10 + self.colors.astype(np.int32))

        # Optional compatibility alias (if some of your code expects y_bin attribute):
        self.y_bin = self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.images[idx]  # [28,28] float32
        y = self.y[idx].astype(np.int32)
        c = int(self.colors[idx])

        rgb = self.COLOR_PALETTE[c]  # [3]
        # Colorize digit foreground: channelwise multiply by the RGB vector.
        x_rgb = (x[None, :, :] * rgb[:, None, None]).astype(np.float32)  # [3,28,28]

        return x_rgb, y


def _as_numpy(batch, dtype):
    """Convert a DataLoader batch to a NumPy array without assuming tensor type."""
    if hasattr(batch, "detach"):
        batch = batch.detach()
    if hasattr(batch, "cpu"):
        batch = batch.cpu()
    return np.asarray(batch, dtype=dtype)


def dataset_to_jax_arrays(
    dataset: Dataset,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    """Load a PyTorch dataset once and convert it to NHWC JAX arrays."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if num_workers > 0 and prefetch_factor < 1:
        raise ValueError("prefetch_factor must be >= 1 when num_workers > 0")

    loader_kwargs = {
        "batch_size": min(int(batch_size), len(dataset)),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(dataset, **loader_kwargs)

    xs, ys = [], []
    for x_batch, y_batch in loader:
        xs.append(_as_numpy(x_batch, np.float32))
        ys.append(_as_numpy(y_batch, np.int32))

    xs = np.concatenate(xs, axis=0)  # NCHW
    xs = np.transpose(xs, (0, 2, 3, 1))  # NHWC
    ys = np.concatenate(ys, axis=0)
    return jnp.array(xs), jnp.array(ys)

DATASET_NUM_CLASSES = {
    "colored_mnist": 2,
    "irm_colored_mnist": 2,
    "standard_mnist_bin": 2,
    "cmnist_ula": 10,
}


def build_dataset(name: str, train: bool, p_corr: float = 0.9, seed: int = 0):
    """Construct a dataset by config name.

    Normalizes constructor signatures across dataset classes (e.g.
    StandardMNISTBin does not accept p_corr/seed).
    """
    factories = {
        "colored_mnist":      lambda: ColoredMNIST(train=train, p_corr=p_corr, seed=seed),
        "irm_colored_mnist":  lambda: IRMColoredMNIST(train=train, p_corr=p_corr, seed=seed),
        "standard_mnist_bin": lambda: StandardMNISTBin(train=train),
        "cmnist_ula":         lambda: CMNISTuLA(train=train, p_corr=p_corr, seed=seed),
    }
    if name not in factories:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {list(factories.keys())}"
        )
    # cmnist_ula test set is always unbiased (uniform colors) per the uLA
    # protocol; p_corr is ignored for the test split.
    if name == "cmnist_ula" and not train and p_corr != 0.1:
        warnings.warn(
            f"cmnist_ula test set is always unbiased (uniform colors); "
            f"p_test={p_corr} is ignored. Set p_test=0.1 to silence this warning."
        )
    return factories[name]()


def make_epoch_batches(x, y, batch_size, seed):
    """
    Fast host-side permutation, device-side gather.
    Avoids jax.random.permutation (slow + sync).

    Note: drops remainder samples (n % batch_size).  Since a fresh random
    permutation is used each epoch, different samples are dropped each time,
    so all samples are seen over multiple epochs.
    """
    n = x.shape[0]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)

    n_trim = (n // batch_size) * batch_size
    perm = perm[:n_trim]

    perm = jnp.asarray(perm)  # move indices to device once

    x_shuf = jnp.take(x, perm, axis=0)
    y_shuf = jnp.take(y, perm, axis=0)

    n_batches = n_trim // batch_size
    xb = x_shuf.reshape((n_batches, batch_size) + x.shape[1:])
    yb = y_shuf.reshape((n_batches, batch_size))
    return xb, yb


def make_eval_batches(x, y, batch_size):
    """Create deterministic (unshuffled) eval batches covering all samples.

    The last batch is zero-padded to ``batch_size``.  Returns ``(xb, yb, counts)``
    where ``counts[i]`` is the number of valid samples in batch ``i``.
    All batches except the last have ``counts[i] == batch_size``.
    """
    n = x.shape[0]
    remainder = n % batch_size
    if remainder != 0:
        pad_n = batch_size - remainder
        x_pad = jnp.zeros((pad_n,) + x.shape[1:], dtype=x.dtype)
        y_pad = jnp.zeros((pad_n,), dtype=y.dtype)
        x = jnp.concatenate([x, x_pad], axis=0)
        y = jnp.concatenate([y, y_pad], axis=0)

    n_padded = x.shape[0]
    n_batches = n_padded // batch_size
    xb = x.reshape((n_batches, batch_size) + x.shape[1:])
    yb = y.reshape((n_batches, batch_size))

    # counts: batch_size for all full batches, remainder for the last
    counts = jnp.full((n_batches,), batch_size, dtype=jnp.int32)
    if remainder != 0:
        counts = counts.at[-1].set(remainder)
    return xb, yb, counts
