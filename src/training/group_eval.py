"""Group-balanced evaluation for spurious correlation benchmarks.

Used by Waterbirds, CelebA, and CivilComments where worst-group accuracy
is the primary evaluation metric.

Also contains MMD performance diagnostics (frozen-feature probes,
per-cell accuracy, offline invariance metrics, weight/batch diagnostics).
"""

import numpy as np
import jax
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


# ---------------------------------------------------------------
# MMD Performance Diagnostics
# ---------------------------------------------------------------

def extract_features(state, x, batch_size):
    """Extract penultimate features z from a deterministic model.

    Args:
        state: model TrainState (uses state.apply_fn and state.params).
        x: [N, ...] input data.
        batch_size: batch size for forward passes.

    Returns:
        [N, D] float32 feature array.
    """
    from src.datasets.datasets import make_eval_batches

    xb, _, counts = make_eval_batches(
        x, jnp.zeros(x.shape[0], dtype=jnp.int32), batch_size,
    )

    @jax.jit
    def _extract(x_batch):
        _, aux = state.apply_fn(
            {"params": state.params}, x_batch, train=False,
        )
        return aux["z"]

    parts = []
    for i in range(xb.shape[0]):
        z = _extract(xb[i])
        parts.append(np.asarray(z[:int(counts[i])]))
    return np.concatenate(parts, axis=0)


# ---- 1. Frozen-feature probes ----

def _linear_probe_accuracy(z_train, y_train, z_test, y_test, num_classes):
    """Train a linear probe (least-squares) and return test accuracy.

    Solves min_W ||Z_train @ W - Y_onehot||^2, predicts argmax(Z_test @ W).
    """
    Y_oh = np.eye(num_classes, dtype=np.float32)[y_train]
    # Add bias column
    Z_aug = np.concatenate([z_train, np.ones((len(z_train), 1), dtype=np.float32)], axis=1)
    W, _, _, _ = np.linalg.lstsq(Z_aug, Y_oh, rcond=None)
    Z_test_aug = np.concatenate([z_test, np.ones((len(z_test), 1), dtype=np.float32)], axis=1)
    preds = np.argmax(Z_test_aug @ W, axis=1)
    return float(np.mean(preds == y_test))


def run_frozen_probes(z, y, s, num_classes, num_colors=None):
    """Run label probe and per-digit conditional color probes on frozen features.

    Uses a 70/30 split of the provided data for train/test of the probes.

    Args:
        z: [N, D] float32 features.
        y: [N] int labels.
        s: [N] int color labels (ground truth).
        num_classes: number of digit classes.
        num_colors: number of colors (defaults to num_classes).

    Returns:
        dict with:
            label_probe_acc: overall label probe accuracy.
            color_probe_acc: average conditional color probe accuracy.
            per_digit_color_probe_acc: dict mapping digit -> color probe acc.
    """
    if num_colors is None:
        num_colors = num_classes

    z = np.asarray(z, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.int32)

    N = len(z)
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    split = int(0.7 * N)
    tr_idx, te_idx = perm[:split], perm[split:]

    # Label probe
    label_acc = _linear_probe_accuracy(
        z[tr_idx], y[tr_idx], z[te_idx], y[te_idx], num_classes,
    )

    # Conditional color probes: for each digit, train s-from-z probe
    per_digit_color_acc = {}
    for d in range(num_classes):
        d_mask_tr = y[tr_idx] == d
        d_mask_te = y[te_idx] == d
        n_tr = int(d_mask_tr.sum())
        n_te = int(d_mask_te.sum())
        if n_tr < num_colors or n_te < 2:
            continue
        acc = _linear_probe_accuracy(
            z[tr_idx][d_mask_tr], s[tr_idx][d_mask_tr],
            z[te_idx][d_mask_te], s[te_idx][d_mask_te],
            num_colors,
        )
        per_digit_color_acc[d] = acc

    color_acc = float(np.mean(list(per_digit_color_acc.values()))) if per_digit_color_acc else 0.0

    return {
        "label_probe_acc": label_acc,
        "color_probe_acc": color_acc,
        "per_digit_color_probe_acc": per_digit_color_acc,
    }


# ---- 2. Per-cell accuracy and confusion ----

def compute_per_cell_accuracy(y_true, y_pred, s, num_classes, num_colors=None):
    """Compute accuracy for each (label, color) cell.

    Args:
        y_true: [N] int true labels.
        y_pred: [N] int predicted labels.
        s: [N] int color labels.
        num_classes: K.
        num_colors: L (defaults to K).

    Returns:
        dict with:
            cell_acc: [K, L] float array (NaN for empty cells).
            cell_count: [K, L] int array.
            worst_cell: (y, s, acc) of worst non-empty cell.
            best_cell: (y, s, acc) of best non-empty cell.
    """
    if num_colors is None:
        num_colors = num_classes

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    s = np.asarray(s, dtype=np.int32)

    cell_acc = np.full((num_classes, num_colors), np.nan)
    cell_count = np.zeros((num_classes, num_colors), dtype=np.int32)

    for yi in range(num_classes):
        for si in range(num_colors):
            mask = (y_true == yi) & (s == si)
            n = int(mask.sum())
            cell_count[yi, si] = n
            if n > 0:
                cell_acc[yi, si] = float(np.mean(y_pred[mask] == yi))

    # Find worst/best non-empty cells
    valid = ~np.isnan(cell_acc)
    worst_idx = np.unravel_index(np.nanargmin(cell_acc), cell_acc.shape) if valid.any() else (0, 0)
    best_idx = np.unravel_index(np.nanargmax(cell_acc), cell_acc.shape) if valid.any() else (0, 0)

    return {
        "cell_acc": cell_acc,
        "cell_count": cell_count,
        "worst_cell": (int(worst_idx[0]), int(worst_idx[1]),
                       float(cell_acc[worst_idx]) if valid.any() else 0.0),
        "best_cell": (int(best_idx[0]), int(best_idx[1]),
                      float(cell_acc[best_idx]) if valid.any() else 0.0),
    }


def compute_per_cell_top_confusion(y_true, y_pred, s, num_classes, num_colors=None):
    """For each (y, s) cell, find the most common wrong prediction.

    Returns:
        dict mapping (y, s) -> (most_common_wrong_class, count, fraction_of_errors).
        Only includes cells with at least 1 error.
    """
    if num_colors is None:
        num_colors = num_classes

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    s = np.asarray(s, dtype=np.int32)

    confusions = {}
    for yi in range(num_classes):
        for si in range(num_colors):
            mask = (y_true == yi) & (s == si)
            wrong = mask & (y_pred != yi)
            n_wrong = int(wrong.sum())
            if n_wrong == 0:
                continue
            wrong_preds = y_pred[wrong]
            counts = np.bincount(wrong_preds, minlength=num_classes)
            top_cls = int(np.argmax(counts))
            confusions[(yi, si)] = (top_cls, int(counts[top_cls]), int(counts[top_cls]) / n_wrong)

    return confusions


# ---- 3. Weight diagnostics ----

def compute_weight_diagnostics(w, y, s, num_classes, num_colors=None):
    """Compute importance weight diagnostics.

    Args:
        w: [N] float importance weights.
        y: [N] int labels.
        s: [N] int colors.
        num_classes: K.

    Returns:
        dict with ESS, max weight, ratio, per-class ESS.
    """
    if num_colors is None:
        num_colors = num_classes

    w = np.asarray(w, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.int32)

    w_sum = w.sum()
    w_sq_sum = (w ** 2).sum()
    ess_global = float(w_sum ** 2 / w_sq_sum) if w_sq_sum > 0 else 0.0

    per_class_ess = {}
    for yi in range(num_classes):
        mask = y == yi
        wc = w[mask]
        if len(wc) == 0:
            continue
        wc_sum = wc.sum()
        wc_sq = (wc ** 2).sum()
        per_class_ess[yi] = float(wc_sum ** 2 / wc_sq) if wc_sq > 0 else 0.0

    median_w = float(np.median(w))
    max_w = float(np.max(w))

    # p(s|y) table
    p_s_given_y = np.zeros((num_classes, num_colors), dtype=np.float64)
    for yi in range(num_classes):
        mask = y == yi
        n_y = mask.sum()
        if n_y == 0:
            continue
        for si in range(num_colors):
            p_s_given_y[yi, si] = float(((y == yi) & (s == si)).sum()) / n_y

    return {
        "ess_global": ess_global,
        "ess_fraction": ess_global / len(w),
        "per_class_ess": per_class_ess,
        "w_max": max_w,
        "w_median": median_w,
        "w_max_over_median": max_w / median_w if median_w > 0 else float("inf"),
        "p_s_given_y": p_s_given_y,
        "n_samples": len(w),
    }


# ---- 4. Batch support statistics ----

def compute_batch_support_stats(y, s, batch_size, num_classes, num_colors=None,
                                n_sample_batches=50, seed=0):
    """Simulate random batches and report (y,s) cell occupancy statistics.

    Args:
        y: [N] int labels (full training set).
        s: [N] int colors (full training set).
        batch_size: batch size used in training.
        num_classes: K.
        n_sample_batches: number of random batches to simulate.
        seed: random seed.

    Returns:
        dict with min/avg/max cell counts, empty cell stats, averaged over batches.
    """
    if num_colors is None:
        num_colors = num_classes

    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.int32)
    N = len(y)
    rng = np.random.RandomState(seed)

    stats = {
        "min_cell": [],
        "avg_cell": [],
        "n_empty": [],
        "n_lt2": [],
    }

    for _ in range(n_sample_batches):
        idx = rng.choice(N, size=batch_size, replace=False)
        yb, sb = y[idx], s[idx]
        cell_counts = np.zeros((num_classes, num_colors), dtype=np.int32)
        for yi in range(num_classes):
            for si in range(num_colors):
                cell_counts[yi, si] = int(((yb == yi) & (sb == si)).sum())
        stats["min_cell"].append(int(cell_counts.min()))
        stats["avg_cell"].append(float(cell_counts.mean()))
        stats["n_empty"].append(int((cell_counts == 0).sum()))
        stats["n_lt2"].append(int((cell_counts < 2).sum()))

    return {
        "min_cell_avg": float(np.mean(stats["min_cell"])),
        "avg_cell_avg": float(np.mean(stats["avg_cell"])),
        "n_empty_avg": float(np.mean(stats["n_empty"])),
        "n_lt2_avg": float(np.mean(stats["n_lt2"])),
        "n_total_cells": num_classes * num_colors,
        "batch_size": batch_size,
    }


# ---- 5. Offline invariance metrics ----

def compute_offline_invariance(z, y, s, num_classes, num_colors=None):
    """Compute full-dataset mean mismatch D_mean between (y,s) cells and class means.

    D_mean = (1/(K*L)) * sum_{y,s} ||mu_{y,s} - mu_y||^2

    Args:
        z: [N, D] features.
        y: [N] int labels.
        s: [N] int colors.
        num_classes: K.

    Returns:
        dict with D_mean, per-class breakdown, cell means shape.
    """
    if num_colors is None:
        num_colors = num_classes

    z = np.asarray(z, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.int32)

    D = z.shape[1]

    # Class means
    class_means = np.zeros((num_classes, D), dtype=np.float64)
    for yi in range(num_classes):
        mask = y == yi
        if mask.sum() > 0:
            class_means[yi] = z[mask].mean(axis=0)

    # Cell means and mismatch
    d_mean_sum = 0.0
    n_valid = 0
    per_class_d_mean = {}

    for yi in range(num_classes):
        class_sum = 0.0
        class_valid = 0
        for si in range(num_colors):
            mask = (y == yi) & (s == si)
            n = int(mask.sum())
            if n == 0:
                continue
            cell_mean = z[mask].mean(axis=0)
            diff = cell_mean - class_means[yi]
            sq_dist = float(np.sum(diff ** 2))
            d_mean_sum += sq_dist
            n_valid += 1
            class_sum += sq_dist
            class_valid += 1
        if class_valid > 0:
            per_class_d_mean[yi] = class_sum / class_valid

    d_mean = d_mean_sum / n_valid if n_valid > 0 else 0.0

    return {
        "d_mean": d_mean,
        "per_class_d_mean": per_class_d_mean,
        "n_valid_cells": n_valid,
    }


# ---- Combined diagnostic runner ----

def run_all_diagnostics(outer_state, x_test, y_test, s_test,
                        num_classes, batch_size,
                        y_train=None, s_train=None, w_train=None):
    """Run all Phase 1 diagnostics and return a combined results dict.

    Args:
        outer_state: trained outer model TrainState.
        x_test: [N_test, ...] test images.
        y_test: [N_test] test labels.
        s_test: [N_test] test ground-truth colors.
        num_classes: K.
        batch_size: batch size for feature extraction.
        y_train: [N_train] train labels (for weight/batch diagnostics).
        s_train: [N_train] train colors (for weight/batch diagnostics).
        w_train: [N_train] importance weights (for weight diagnostics).

    Returns:
        dict with all diagnostic results.
    """
    results = {}

    # Extract test features
    z_test = extract_features(outer_state, x_test, batch_size)
    y_np = np.asarray(y_test, dtype=np.int32)
    s_np = np.asarray(s_test, dtype=np.int32)

    # Predictions from the model
    y_pred = np.argmax(
        np.asarray(z_test),  # z is penultimate; need logits for predictions
        axis=1,
    )
    # Actually we need logits, not z. Let me get predictions properly.
    from src.datasets.datasets import make_eval_batches

    xb, _, counts = make_eval_batches(x_test, y_test, batch_size)

    @jax.jit
    def _predict(x_batch):
        logits, _ = outer_state.apply_fn(
            {"params": outer_state.params}, x_batch, train=False,
        )
        return jnp.argmax(logits, axis=-1)

    pred_parts = []
    for i in range(xb.shape[0]):
        p = _predict(xb[i])
        pred_parts.append(np.asarray(p[:int(counts[i])]))
    y_pred = np.concatenate(pred_parts, axis=0)

    # 1. Frozen-feature probes
    probes = run_frozen_probes(z_test, y_np, s_np, num_classes)
    results.update({f"probe/{k}": v for k, v in probes.items()
                    if not isinstance(v, dict)})
    results["probe/per_digit_color_probe_acc"] = probes["per_digit_color_probe_acc"]

    # 2. Per-cell accuracy
    cell_results = compute_per_cell_accuracy(y_np, y_pred, s_np, num_classes)
    results["cell/worst"] = cell_results["worst_cell"]
    results["cell/best"] = cell_results["best_cell"]
    results["cell/acc_matrix"] = cell_results["cell_acc"]
    results["cell/count_matrix"] = cell_results["cell_count"]

    # Per-cell confusions
    confusions = compute_per_cell_top_confusion(y_np, y_pred, s_np, num_classes)
    results["cell/top_confusions"] = confusions

    # 5. Offline invariance
    inv = compute_offline_invariance(z_test, y_np, s_np, num_classes)
    results["invariance/d_mean"] = inv["d_mean"]
    results["invariance/per_class_d_mean"] = inv["per_class_d_mean"]

    # 3. Weight diagnostics (if training data provided)
    if w_train is not None and y_train is not None and s_train is not None:
        wd = compute_weight_diagnostics(
            np.asarray(w_train), np.asarray(y_train),
            np.asarray(s_train), num_classes,
        )
        results["weights/ess_global"] = wd["ess_global"]
        results["weights/ess_fraction"] = wd["ess_fraction"]
        results["weights/w_max"] = wd["w_max"]
        results["weights/w_max_over_median"] = wd["w_max_over_median"]
        results["weights/p_s_given_y"] = wd["p_s_given_y"]
        results["weights/per_class_ess"] = wd["per_class_ess"]

    # 4. Batch support stats (if training data provided)
    if y_train is not None and s_train is not None:
        bs = compute_batch_support_stats(
            np.asarray(y_train), np.asarray(s_train),
            batch_size, num_classes,
        )
        results["batch/n_empty_avg"] = bs["n_empty_avg"]
        results["batch/n_lt2_avg"] = bs["n_lt2_avg"]
        results["batch/avg_cell_avg"] = bs["avg_cell_avg"]
        results["batch/min_cell_avg"] = bs["min_cell_avg"]

    return results


def print_diagnostics(diag, num_classes=10):
    """Pretty-print diagnostic results to stdout."""
    print("\n" + "=" * 70)
    print("  MMD PERFORMANCE DIAGNOSTICS")
    print("=" * 70)

    # Probes
    print(f"\n  Label probe accuracy:             {diag.get('probe/label_probe_acc', 0):.4f}")
    print(f"  Conditional color probe accuracy:  {diag.get('probe/color_probe_acc', 0):.4f}")
    chance = 1.0 / num_classes
    print(f"  (chance level = {chance:.3f})")

    per_digit = diag.get("probe/per_digit_color_probe_acc", {})
    if per_digit:
        print("  Per-digit color probe:")
        for d in sorted(per_digit):
            print(f"    digit {d}: {per_digit[d]:.4f}")

    # Interpretation
    label_acc = diag.get("probe/label_probe_acc", 0)
    color_acc = diag.get("probe/color_probe_acc", 0)
    if label_acc > 0.95 and color_acc > chance + 0.05:
        print("  --> DIAGNOSIS: Failure A (color still leaking)")
    elif label_acc < 0.90 and color_acc <= chance + 0.03:
        print("  --> DIAGNOSIS: Failure B (over-compression)")
    elif label_acc < 0.90 and color_acc > chance + 0.05:
        print("  --> DIAGNOSIS: Failure C (poor optimization)")
    elif label_acc > 0.95 and color_acc <= chance + 0.03:
        print("  --> DIAGNOSIS: Representation looks good")

    # Per-cell accuracy
    cell_acc = diag.get("cell/acc_matrix")
    if cell_acc is not None:
        print(f"\n  Per-cell accuracy (10x10 grid, y=row, s=col):")
        header = "       " + "".join(f"  s={si:<3d}" for si in range(min(num_classes, cell_acc.shape[1])))
        print(header)
        for yi in range(min(num_classes, cell_acc.shape[0])):
            row = f"  y={yi}: "
            for si in range(min(num_classes, cell_acc.shape[1])):
                v = cell_acc[yi, si]
                if np.isnan(v):
                    row += "   --- "
                else:
                    row += f"  {v:.3f}"
            print(row)

        worst = diag.get("cell/worst", (0, 0, 0))
        best = diag.get("cell/best", (0, 0, 0))
        print(f"  Worst cell: (y={worst[0]}, s={worst[1]}) acc={worst[2]:.4f}")
        print(f"  Best cell:  (y={best[0]}, s={best[1]}) acc={best[2]:.4f}")

    # Top confusions
    confusions = diag.get("cell/top_confusions", {})
    if confusions:
        print(f"\n  Top confusions (cells with errors):")
        sorted_conf = sorted(confusions.items(), key=lambda kv: -kv[1][1])
        for (yi, si), (wrong_cls, cnt, frac) in sorted_conf[:15]:
            print(f"    (y={yi}, s={si}) -> predicted {wrong_cls} "
                  f"({cnt} errors, {frac:.0%} of cell errors)")

    # Invariance
    d_mean = diag.get("invariance/d_mean")
    if d_mean is not None:
        print(f"\n  Offline D_mean (feature mean mismatch): {d_mean:.6f}")

    # Weight diagnostics
    ess = diag.get("weights/ess_global")
    if ess is not None:
        n = diag.get("weights/n_samples", 0) if "weights/n_samples" in diag else 0
        print(f"\n  Weight diagnostics:")
        print(f"    ESS global:       {ess:.0f} / {diag.get('weights/ess_fraction', 0):.2%} of N")
        print(f"    w_max:            {diag.get('weights/w_max', 0):.4f}")
        print(f"    w_max/median:     {diag.get('weights/w_max_over_median', 0):.2f}")

    # Batch support
    n_empty = diag.get("batch/n_empty_avg")
    if n_empty is not None:
        total = diag.get("batch/n_total_cells", num_classes ** 2)
        print(f"\n  Batch support (avg over 50 random batches):")
        print(f"    Empty cells:      {n_empty:.1f} / {num_classes**2}")
        print(f"    Cells with <2:    {diag.get('batch/n_lt2_avg', 0):.1f} / {num_classes**2}")
        print(f"    Avg cell count:   {diag.get('batch/avg_cell_avg', 0):.2f}")

    print("=" * 70 + "\n")
