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
