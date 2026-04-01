"""Streaming data loader for datasets too large to materialize.

Wraps a PyTorch DataLoader and yields (x_jax_nhwc, y_jax) batches.
Used by CelebA and CivilComments where full materialization is infeasible.
"""

import numpy as np
import jax.numpy as jnp
from torch.utils.data import DataLoader


class StreamingLoader:
    """Streaming data loader that yields JAX arrays.

    Args:
        dataset: PyTorch Dataset.
        batch_size: batch size.
        shuffle: whether to shuffle.
        drop_last: whether to drop the last incomplete batch.
        num_workers: PyTorch DataLoader workers.
        to_nhwc: if True, transpose from CHW to NHWC (for image datasets).
    """

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=True,
                 num_workers=0, to_nhwc=True):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        self.to_nhwc = to_nhwc

    def __iter__(self):
        for x_batch, y_batch in self.loader:
            x = np.array(x_batch, dtype=np.float32)
            if self.to_nhwc and x.ndim == 4:
                x = np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
            x = jnp.array(x)
            y = jnp.array(np.array(y_batch), dtype=jnp.int32)
            yield x, y

    def __len__(self):
        return len(self.loader)


def streaming_train_epoch(state, loader, train_step_fn, rng, lamb, alpha):
    """Run one training epoch using a streaming data loader.

    Uses a Python for-loop instead of lax.scan (data not pre-materialized).

    Returns:
        (state, avg_metrics) where avg_metrics is a dict of scalar averages.
    """
    import jax

    metrics_sum = None
    n_batches = 0

    for x_batch, y_batch in loader:
        rng, step_rng = jax.random.split(rng)
        state, metrics = train_step_fn(state, (x_batch, y_batch), step_rng, lamb, alpha)

        if metrics_sum is None:
            metrics_sum = metrics
        else:
            metrics_sum = jax.tree.map(lambda a, b: a + b, metrics_sum, metrics)
        n_batches += 1

    if n_batches == 0:
        return state, {}

    avg_metrics = jax.tree.map(lambda m: m / n_batches, metrics_sum)
    return state, avg_metrics


def streaming_eval_epoch(state, loader, eval_step_fn, rng):
    """Run one eval epoch using a streaming data loader.

    Returns:
        (avg_loss, avg_acc) as floats.
    """
    import jax

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x_batch, y_batch in loader:
        rng, step_rng = jax.random.split(rng)
        loss, acc = eval_step_fn(state, (x_batch, y_batch), step_rng)
        total_loss += float(loss)
        total_acc += float(acc)
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0

    return total_loss / n_batches, total_acc / n_batches
