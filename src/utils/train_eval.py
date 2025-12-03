import jax.numpy as jnp
from src.utils.cfg import CFG
from src.pure_fns.losses_metrics import cross_entropy_loss, sum_of_squared_params, accuracy
from src.utils.data.load_data import to_jax_batch



def evaluate_model(state, data_loader, *, tqdm_str: str = ''):
    """
    Aggregate metrics over a DataLoader without doing backward passes.

    Returns a dict with:
      - 'loss': mean cross-entropy (nats)
      - 'l2': L2 penalty scaled by CFG.wd (single value, same for all batches)
      - 'l2_raw': unscaled L2 sum of squares (for completeness)
      - 'acc': mean accuracy
      - 'I_ry': MI(R;Y) lower bound in nats (H(Y) - CE), using model logits
      - 'H_y': label entropy in nats
      - 'counts': per-class counts (jnp.ndarray)
      - 'num_batches': number of evaluated batches
      - 'num_examples': total number of evaluated examples
    """
    # L2 penalty does not depend on data; compute once.
    l2_raw = sum_of_squared_params(state.params)
    l2_scaled = l2_raw * CFG.wd

    ce_vals = []
    acc_vals = []
    y_counts = jnp.zeros(CFG.num_classes, dtype=jnp.int64)
    n_examples = 0
    n_batches = 0

    iterator = data_loader
    if tqdm_str:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"Eval[{tqdm_str}]", leave=False)

    for images_t, labels_t in iterator:
        xb, yb = to_jax_batch(images_t, labels_t)
        logits = state.apply_fn({"params": state.params}, xb, train=False)

        # accuracy (jitted)
        acc = accuracy(logits, yb)
        acc_vals.append(float(acc))

        # CE via logits (same as your train loss definition)
        ce = cross_entropy_loss(logits, yb)
        ce_vals.append(float(ce))

        # label histogram for H(Y)
        batch_counts = jnp.bincount(yb, length=CFG.num_classes)
        y_counts = y_counts + batch_counts

        n_examples += xb.shape[0]
        n_batches += 1

    # Aggregate
    import numpy as np
    mean_ce = float(np.mean(ce_vals)) if ce_vals else 0.0
    mean_acc = float(np.mean(acc_vals)) if acc_vals else 0.0

    # H(Y) in nats (ignore zero bins)
    counts_np = np.asarray(y_counts)
    total = int(counts_np.sum()) or 1
    p = counts_np / total
    p_nz = p[p > 0]
    H_y = float(-(p_nz * np.log(p_nz)).sum())

    # MI lower bound in bits: H(Y) - CE
    I_ry = H_y - mean_ce

    return {
        "loss": mean_ce,
        "l2": l2_scaled,
        "l2_raw": l2_raw,
        "acc": mean_acc,
        "I_ry": I_ry,
        "H_y": H_y,
        "counts": y_counts,
        "num_batches": n_batches,
        "num_examples": n_examples,
    }


def calc_loss_and_metrics_for_batch(state, xb, yb, train_step=None, eval_step=None):
    """
    Return (state, metrics) where metrics has:
      - 'loss' : CE loss (nats)
      - 'l2'   : L2 penalty term
      - 'acc'  : accuracy
      - 'ce'   : same as 'loss' for clarity
      - 'counts': per-class label histogram
    """

    assert train_step or eval_step, 'Need either train step or eval step'

    if train_step:
        new_state, loss, acc = train_step(state, xb, yb)
        batch_counts = jnp.bincount(yb, length=CFG.num_classes)
        l2_val = 0.0 if not CFG.wd else sum_of_squared_params(new_state.params) * CFG.wd
        metrics = {
            "loss": float(loss),
            "l2":   l2_val,
            "acc":  float(acc),
            "ce":   float(loss),
            "y_counts": batch_counts,
        }
        return new_state, metrics

    else:  # eval/test
        acc = eval_step(state.params, xb, yb)
        logits = state.apply_fn({"params": state.params}, xb, train=False)
        ce = cross_entropy_loss(logits, yb)
        batch_counts = jnp.bincount(yb, length=CFG.num_classes)
        l2_val = 0.0 if not CFG.wd else sum_of_squared_params(state.params) * CFG.wd
        metrics = {
            "loss": float(ce),
            "l2":   l2_val,
            "acc":  float(acc),
            "ce":   float(ce),
            "y_counts": batch_counts,
        }
        return state, metrics