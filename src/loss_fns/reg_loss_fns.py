# ============================================================
# 3) Regularization Penalties (Spectral / HSIC placeholders)
# ============================================================

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


def linear_barrier(R_eff, eps=1e-10):
    return R_eff + (1 / (R_eff - 1 + eps))


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
    z1 = z1.reshape((z1.shape[0], -1))
    z2 = z2.reshape((z2.shape[0], -1))

    # standardize per-dim for kernel stability
    z1 = (z1 - z1.mean(axis=0, keepdims=True)) / (z1.std(axis=0, keepdims=True) + 1e-6)
    z2 = (z2 - z2.mean(axis=0, keepdims=True)) / (z2.std(axis=0, keepdims=True) + 1e-6)

    K = center_gram(rbf_gram(z1))
    L = center_gram(rbf_gram(z2))

    n = z1.shape[0]
    return jnp.sum(K * L) / ((n - 1.0) ** 2)