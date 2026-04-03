"""Group-balanced evaluation for spurious correlation benchmarks.

Used by Waterbirds, CelebA, and CivilComments where worst-group accuracy
is the primary evaluation metric.
"""

import jax.numpy as jnp


def eval_group_accuracies(predictions, labels, groups):
    """Compute per-group and worst-group accuracy.

    Args:
        predictions: (N,) int array of predicted class labels.
        labels: (N,) int array of true class labels.
        groups: (N,) int array of group assignments.

    Returns:
        dict with keys:
            avg_acc: overall accuracy.
            worst_group_acc: minimum per-group accuracy.
            worst_group_id: group id with lowest accuracy.
            per_group_acc: dict mapping group_id -> accuracy.
            per_group_count: dict mapping group_id -> sample count.
    """
    correct = (predictions == labels)
    unique_groups = jnp.unique(groups)

    per_group_acc = {}
    per_group_count = {}
    worst_acc = 1.0
    worst_id = -1

    for g in unique_groups:
        g_int = int(g)
        mask = (groups == g)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(correct[mask].mean())
        per_group_acc[g_int] = acc
        per_group_count[g_int] = count
        if acc < worst_acc:
            worst_acc = acc
            worst_id = g_int

    return {
        "avg_acc": float(correct.mean()),
        "worst_group_acc": worst_acc,
        "worst_group_id": worst_id,
        "per_group_acc": per_group_acc,
        "per_group_count": per_group_count,
    }
