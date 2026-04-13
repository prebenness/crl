# ============================================================
# 3) Regularization Penalties (Spectral / HSIC placeholders)
# ============================================================

import jax
import jax.numpy as jnp
import jax.lax as lax


def svd_spectral_penalty(z):
    """
    Computes the nuclear norm (sum of singular values) of the batch.
    Encourages the representation to lie on a low-dimensional manifold.
    """
    # Center the batch (conceptually important for covariance rank)
    z_centered = z - jnp.mean(z, axis=0, keepdims=True)
    
    # Compute singular values (SVD)
    # full_matrices=False is critical for performance
    _, s, _ = jnp.linalg.svd(z_centered, full_matrices=False)
    
    # Minimize the sum of singular values (nuclear norm)
    return jnp.sum(s)


def frobenius_penalty(z):
    """
    Fast surrogate for nuclear norm:

    - Center z over the batch
    - Compute covariance C = (Zᵀ Z) / N
    - Penalise ||C||_F² = sum_ij C_ij²

    This is cheap (just matmuls) and JAX-friendly,
    but still encourages low-rank / low-variance embeddings.
    """
    # z: [batch, dim]
    z = z.astype(jnp.float32)

    # Center across the batch
    z_centered = z - jnp.mean(z, axis=0, keepdims=True)

    # Covariance (up to a constant factor)
    n = z_centered.shape[0]
    cov = (z_centered.T @ z_centered) / n  # [dim, dim]

    # Matrix-wide average of squared cov values - essentially a normalised frobenius norm
    return jnp.mean(cov ** 2)


def effective_rank_penalty(z, eps=1e-10):
    """
    Scale-invariant, rank-focused penalty for the bottleneck Z.

    z: [batch, dim] representation.

    Computes R_eff = (trace(C)^2 / ||C||_F^2), where C is the
    centered covariance. R_eff is in [1, dim] and behaves like
    an "effective rank": 1 when all variance is in one dimension,
    dim when variance is spread equally.

    Minimizing this encourages low effective rank while being
    invariant to global rescaling of z.
    """
    # Center across batch
    zc = z - jnp.mean(z, axis=0, keepdims=True)

    # Covariance (up to constant factor)
    n = zc.shape[0]
    C = (zc.T @ zc) / n  # [dim, dim]

    # trace and Frobenius norm squared
    trace = jnp.trace(C)
    frob_sq = jnp.sum(C ** 2)

    # Effective-rank-like quantity
    R_eff = (trace ** 2) / (frob_sq + eps)

    return R_eff


def init_ema_state(num_classes, z_dim):
    """Initialize EMA state for mean matching."""
    K, D = num_classes, z_dim
    return {
        "means": jnp.zeros((K, K, D), dtype=jnp.float32),
        "valid": jnp.zeros((K, K), dtype=jnp.float32),
    }


def mean_match_loss(z, y, s, ema_state, num_classes, ema_alpha, eps=1e-8):
    """EMA mean matching invariance loss.

    For each (y,s) cell present in the batch, computes the batch cell mean
    and penalizes its deviation from the EMA class mean (stop-gradiented).
    Then updates the EMA with the batch cell means.

    Returns (loss, updated_ema_state).
    """
    K = num_classes
    D = z.shape[-1]

    # Compute per-cell batch means
    y_oh = jax.nn.one_hot(y, K)                     # [B, K]
    s_oh = jax.nn.one_hot(s, K)                     # [B, K]
    cell_mask = y_oh[:, :, None] * s_oh[:, None, :]  # [B, K, K]
    cell_sum = jnp.einsum("bcs,bd->csd", cell_mask, z)  # [K, K, D]
    cell_count = cell_mask.sum(axis=0)                # [K, K]
    cell_present = (cell_count > 0).astype(jnp.float32)  # [K, K]
    batch_cell_mean = cell_sum / (cell_count[..., None] + eps)  # [K, K, D]

    # Class means from EMA (average over all valid colors for each class)
    ema_valid = ema_state["valid"]                     # [K, K]
    ema_means = ema_state["means"]                     # [K, K, D]
    valid_count = ema_valid.sum(axis=1, keepdims=True) + eps  # [K, 1]
    class_mean_ema = (ema_means * ema_valid[..., None]).sum(axis=1) / valid_count  # [K, D]

    # Loss: deviation of batch cell means from class mean (stop-grad target)
    target = lax.stop_gradient(class_mean_ema)         # [K, D]
    deviation = batch_cell_mean - target[:, None, :]   # [K, K, D]
    sq_dev = (deviation ** 2).sum(axis=-1)             # [K, K]
    loss = (sq_dev * cell_present).sum() / (cell_present.sum() + eps)

    # EMA update (stop-gradient on batch means for storage)
    sg_batch = lax.stop_gradient(batch_cell_mean)
    updated_means = jnp.where(
        cell_present[..., None] > 0,
        ema_alpha * ema_means + (1 - ema_alpha) * sg_batch,
        ema_means,
    )
    updated_valid = jnp.maximum(ema_valid, cell_present)

    new_ema = {"means": updated_means, "valid": updated_valid}
    return loss, new_ema


def _standardize(z: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Per-dimension z-score standardization for kernel stability."""
    return (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + eps)


def center_gram(K: jnp.ndarray) -> jnp.ndarray:
    mean_row = jnp.mean(K, axis=0, keepdims=True)
    mean_col = jnp.mean(K, axis=1, keepdims=True)
    mean_all = jnp.mean(K)
    return K - mean_row - mean_col + mean_all


def rbf_gram(X: jnp.ndarray, sigma2: jnp.ndarray | None = None) -> jnp.ndarray:
    x2 = jnp.sum(X * X, axis=1, keepdims=True)
    d2 = x2 + x2.T - 2.0 * (X @ X.T)
    d2 = jnp.maximum(d2, 0.0)

    if sigma2 is None:
        B = X.shape[0]
        iu = jnp.triu_indices(B, k=1)
        med = jnp.median(lax.stop_gradient(d2[iu]))
        sigma2 = med + 1e-6

    return jnp.exp(-d2 / (2.0 * sigma2))


def hsic_rbf(z1: jnp.ndarray, z2: jnp.ndarray) -> jnp.ndarray:
    z1 = _standardize(z1.reshape((z1.shape[0], -1)))
    z2 = _standardize(z2.reshape((z2.shape[0], -1)))

    K = center_gram(rbf_gram(z1))
    L = center_gram(rbf_gram(z2))

    n = z1.shape[0]
    return jnp.sum(K * L) / ((n - 1.0) ** 2)

def class_cond_mmd_rbf(
    z: jnp.ndarray,
    y: jnp.ndarray,
    s: jnp.ndarray,
    num_classes: int | None = None,
    num_colors: int | None = None,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Class-conditional all-pairs MMD penalty.

    Computes the conditional invariance penalty from Eq. 8 of the note:
        R_inv = mean over valid pairs of MMD^2(z|y,s ; z|y,s')
    for all classes y and all color pairs (s, s') with s < s'.

    Args:
        z: [B, D] representation from the outer model.
        y: [B] integer class labels.
        s: [B] integer predicted color labels (from inner model argmax).
        num_classes: number of classes (static for JIT).
        num_colors: number of color values (static for JIT).
        eps: numerical stability constant.

    Returns:
        Scalar penalty (0 when no valid pairs exist).
    """
    z = _standardize(z.reshape((z.shape[0], -1)), eps=eps)
    y = y.reshape((-1,))
    s = s.reshape((-1,))

    K_full = rbf_gram(z)
    B = z.shape[0]

    classes = jnp.arange(num_classes)
    colors = jnp.arange(num_colors)

    # Build all (s, s') pairs with s < s'
    pair_s = []
    pair_sp = []
    for i in range(num_colors):
        for j in range(i + 1, num_colors):
            pair_s.append(i)
            pair_sp.append(j)
    n_pairs = len(pair_s)
    pair_s_arr = jnp.array(pair_s, dtype=jnp.int32)
    pair_sp_arr = jnp.array(pair_sp, dtype=jnp.int32)

    def mmd_for_class_pair(c, si, sj):
        """MMD^2 between z[y=c, s=si] and z[y=c, s=sj]."""
        m_a = ((y == c) & (s == si)).astype(K_full.dtype)
        m_b = ((y == c) & (s == sj)).astype(K_full.dtype)
        n_a = jnp.sum(m_a)
        n_b = jnp.sum(m_b)

        M_aa = m_a[:, None] * m_a[None, :]
        M_bb = m_b[:, None] * m_b[None, :]
        M_ab = m_a[:, None] * m_b[None, :]

        mean_aa = jnp.sum(K_full * M_aa) / (n_a * n_a + eps)
        mean_bb = jnp.sum(K_full * M_bb) / (n_b * n_b + eps)
        mean_ab = jnp.sum(K_full * M_ab) / (n_a * n_b + eps)

        mmd_sq = mean_aa + mean_bb - 2.0 * mean_ab
        valid = ((n_a >= 2.0) & (n_b >= 2.0)).astype(K_full.dtype)
        return mmd_sq * valid, valid

    def mmd_for_class(c):
        """Sum over all color pairs for class c."""
        def per_pair(idx):
            return mmd_for_class_pair(c, pair_s_arr[idx], pair_sp_arr[idx])

        vals, valids = jax.vmap(per_pair)(jnp.arange(n_pairs))
        return jnp.sum(vals), jnp.sum(valids)

    all_vals, all_counts = jax.vmap(mmd_for_class)(classes)
    total_valid = jnp.sum(all_counts)
    return jnp.sum(all_vals) / (total_valid + eps)


def init_mmd_bank(num_classes, bank_size, x_shape):
    """Initialize an empty FIFO memory bank for class-conditional MMD.

    Stores raw inputs x so that z can be recomputed from the current encoder.

    Returns a dict with:
        x:   [K*K*Q, *x_shape]  stored inputs (zeros)
        y:   [K*K*Q]            class labels (-1 = empty)
        s:   [K*K*Q]            color labels (-1 = empty)
        ptr: [K, K]             next write position per (y,s) cell
    """
    K, Q = num_classes, bank_size
    total = K * K * Q
    return {
        "x": jnp.zeros((total,) + tuple(x_shape), dtype=jnp.float32),
        "y": jnp.full((total,), -1, dtype=jnp.int32),
        "s": jnp.full((total,), -1, dtype=jnp.int32),
        "ptr": jnp.zeros((K, K), dtype=jnp.int32),
    }


def update_mmd_bank(bank, x, y, s, num_classes, bank_size):
    """FIFO update: push current batch's (x, y, s) into the bank.

    Vectorized scatter — computes per-sample within-cell offsets to avoid
    collisions, then writes all samples in one scatter operation.
    """
    K, Q = num_classes, bank_size
    B = x.shape[0]

    # Compute within-cell offsets: for sample i, how many earlier samples
    # in this batch share the same (y, s) cell?
    cell_id = y * K + s  # [B]
    same_cell = (cell_id[:, None] == cell_id[None, :])  # [B, B]
    lower = jnp.tril(jnp.ones((B, B), dtype=jnp.bool_), k=-1)
    offsets = jnp.sum(same_cell & lower, axis=1)  # [B]

    # Compute flat write indices
    ptr_base = bank["ptr"][y, s]  # [B]
    slot = (ptr_base + offsets) % Q  # [B]
    flat_idx = y * (K * Q) + s * Q + slot  # [B]

    # Scatter writes
    new_x = bank["x"].at[flat_idx].set(x)
    new_y = bank["y"].at[flat_idx].set(y)
    new_s = bank["s"].at[flat_idx].set(s)

    # Advance pointers by per-cell batch counts
    y_oh = jax.nn.one_hot(y, K)  # [B, K]
    s_oh = jax.nn.one_hot(s, K)  # [B, K]
    cell_counts = (y_oh.T @ s_oh).astype(jnp.int32)  # [K, K]
    new_ptr = bank["ptr"] + cell_counts

    return {"x": new_x, "y": new_y, "s": new_s, "ptr": new_ptr}


def _weighted_center_gram(K: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted centering: H K H^T with H = I - 1 w^T, where sum(w)=1.
    w: [B] (can include zeros for masked entries)
    """
    B = K.shape[0]
    ones = jnp.ones((B,), dtype=K.dtype)

    # row/col "means" under weights
    row_mean = w @ K           # [B]
    col_mean = K @ w           # [B]
    mean_all = w @ (K @ w)     # scalar

    Kc = K - jnp.outer(ones, row_mean) - jnp.outer(col_mean, ones) + mean_all
    return Kc


def class_cond_hsic_rbf(
    z1: jnp.ndarray,
    z2: jnp.ndarray,
    y: jnp.ndarray,
    num_classes: int | None = None,
    weight_by_freq: bool = False,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Class-conditional HSIC: average over classes c of HSIC(z1,z2 | y=c).

    Signature matches hsic_rbf(z1,z2) except it additionally takes y.
    (Optionally pass num_classes to avoid dynamic shapes under jit.)

    - z1, z2: [B, D1], [B, D2] (or any shape with leading batch dim)
    - y: [B] integer labels
    - Returns: scalar penalty
    """
    z1 = _standardize(z1.reshape((z1.shape[0], -1)), eps=eps)
    z2 = _standardize(z2.reshape((z2.shape[0], -1)), eps=eps)
    y = y.reshape((-1,))

    # Precompute full-batch kernels once (O(B^2)), then mask per class.
    K_full = rbf_gram(z1)
    L_full = rbf_gram(z2)

    classes = jnp.arange(num_classes)

    def hsic_for_class(c):
        m = (y == c).astype(K_full.dtype)           # [B]
        n_c = jnp.sum(m)                            # scalar

        # Mask to the class submatrix
        M = m[:, None] * m[None, :]                 # [B,B]
        K = K_full * M
        L = L_full * M

        # If n_c < 2, return 0 (no meaningful HSIC)
        def compute():
            # uniform weights over members (sum to 1)
            w = m / (n_c + eps)                     # [B]

            Kc = _weighted_center_gram(K, w) * M    # re-mask to keep nonmembers exactly 0
            Lc = _weighted_center_gram(L, w) * M

            denom = (n_c - 1.0) ** 2 + eps
            return jnp.sum(Kc * Lc) / denom

        return lax.cond(n_c >= 2.0, compute, lambda: jnp.array(0.0, dtype=K_full.dtype)), n_c

    hsic_vals, counts = jax.vmap(hsic_for_class)(classes)  # [C], [C]
    valid = (counts >= 2.0).astype(hsic_vals.dtype)

    def reduce_unweighted():
        denom = jnp.sum(valid) + eps
        return jnp.sum(hsic_vals * valid) / denom

    def reduce_freq_weighted():
        # Weight classes by their batch frequency among valid classes
        w = (counts * valid)
        w = w / (jnp.sum(w) + eps)
        return jnp.sum(hsic_vals * w)

    return lax.cond(weight_by_freq, reduce_freq_weighted, reduce_unweighted)
