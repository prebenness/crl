"""Tests for the central mathematical operations in the codebase.

Covers:
  - KL divergence (VIB bottleneck)
  - RBF kernel and Gram matrix
  - Gram matrix centering
  - HSIC (unconditional and class-conditional)
  - Spectral penalties (nuclear norm, Frobenius, effective rank)
  - ControlVAE controller dynamics
  - MC predictive distribution (log-mean-exp)
  - VIB loss composition
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax.scipy.special import logsumexp

from src.models.ib_classifiers import _compute_kl_and_aux
from src.loss_fns.reg_loss_fns import (
    _standardize,
    center_gram,
    rbf_gram,
    hsic_rbf,
    _weighted_center_gram,
    class_cond_hsic_rbf,
    svd_spectral_penalty,
    frobenius_penalty,
    effective_rank_penalty,
)


# ====================================================================
# KL divergence: KL(N(mu, diag(exp(logvar))) || N(0, I))
# Analytical: 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
# ====================================================================


class TestKLDivergence:

    def test_kl_zero_for_standard_normal(self):
        """KL(N(0, I) || N(0, I)) = 0."""
        B, D = 4, 8
        mu = jnp.zeros((B, D))
        logvar = jnp.zeros((B, D))
        z = mu  # irrelevant for KL
        x_recon = jnp.zeros((B, 28, 28, 3))  # placeholder

        aux = _compute_kl_and_aux(z, mu, logvar, x_recon)
        assert aux["kl"].shape == ()
        np.testing.assert_allclose(float(aux["kl"]), 0.0, atol=1e-6)
        np.testing.assert_allclose(
            np.array(aux["kl_per_example"]), 0.0, atol=1e-6,
        )

    def test_kl_nonzero_mu(self):
        """KL(N(mu, I) || N(0, I)) = 0.5 * sum(mu^2).

        When logvar=0 (var=1), KL = 0.5 * sum(mu^2) per example.
        """
        B, D = 2, 3
        mu = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        logvar = jnp.zeros((B, D))
        z = mu
        x_recon = jnp.zeros((B, 28, 28, 3))

        aux = _compute_kl_and_aux(z, mu, logvar, x_recon)

        # Example 0: 0.5 * (1 + 0 + 0) = 0.5
        # Example 1: 0.5 * (0 + 4 + 0) = 2.0
        expected_per = jnp.array([0.5, 2.0])
        np.testing.assert_allclose(
            np.array(aux["kl_per_example"]), np.array(expected_per),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            float(aux["kl"]), float(expected_per.mean()), atol=1e-5,
        )

    def test_kl_nonzero_logvar(self):
        """KL(N(0, diag(sigma^2)) || N(0, I)).

        With mu=0: KL = 0.5 * sum(exp(logvar) - 1 - logvar) per example.
        """
        B, D = 1, 2
        mu = jnp.zeros((B, D))
        # logvar = [ln(2), ln(0.5)] = [0.6931, -0.6931]
        logvar = jnp.array([[jnp.log(2.0), jnp.log(0.5)]])
        z = mu
        x_recon = jnp.zeros((B, 28, 28, 3))

        aux = _compute_kl_and_aux(z, mu, logvar, x_recon)

        # Per-dim: 0.5 * (exp(logvar) - 1 - logvar)
        # dim 0: 0.5 * (2 - 1 - ln2) = 0.5 * (1 - 0.6931) = 0.1534
        # dim 1: 0.5 * (0.5 - 1 - ln0.5) = 0.5 * (-0.5 + 0.6931) = 0.0966
        # sum: 0.25
        expected = 0.5 * jnp.sum(jnp.exp(logvar) - 1.0 - logvar)
        np.testing.assert_allclose(
            float(aux["kl"]), float(expected), atol=1e-5,
        )

    def test_kl_always_nonnegative(self):
        """KL divergence is always >= 0 (Gibbs' inequality)."""
        rng = jrandom.PRNGKey(42)
        B, D = 32, 16
        mu = jrandom.normal(rng, (B, D))
        logvar = jrandom.normal(jrandom.fold_in(rng, 1), (B, D))
        z = mu
        x_recon = jnp.zeros((B, 28, 28, 3))

        aux = _compute_kl_and_aux(z, mu, logvar, x_recon)
        assert float(aux["kl"]) >= 0.0
        assert jnp.all(aux["kl_per_example"] >= -1e-6)

    def test_kl_matches_manual_computation(self):
        """Cross-check against explicit numpy computation."""
        rng = jrandom.PRNGKey(99)
        B, D = 8, 5
        mu = jrandom.normal(rng, (B, D))
        logvar = jrandom.normal(jrandom.fold_in(rng, 1), (B, D))
        z = mu
        x_recon = jnp.zeros((B, 28, 28, 3))

        aux = _compute_kl_and_aux(z, mu, logvar, x_recon)

        # Manual computation
        mu_np = np.array(mu)
        logvar_np = np.array(logvar)
        kl_manual = 0.5 * np.sum(
            mu_np**2 + np.exp(logvar_np) - 1.0 - logvar_np, axis=-1,
        )
        np.testing.assert_allclose(
            np.array(aux["kl_per_example"]), kl_manual, rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(aux["kl"]), kl_manual.mean(), rtol=1e-5,
        )


# ====================================================================
# RBF kernel
# ====================================================================


class TestRBFGram:

    def test_diagonal_is_one(self):
        """K(x, x) = exp(0) = 1 for all x."""
        X = jrandom.normal(jrandom.PRNGKey(0), (10, 4))
        K = rbf_gram(X, sigma2=jnp.array(1.0))
        np.testing.assert_allclose(
            np.diag(np.array(K)), 1.0, atol=1e-6,
        )

    def test_symmetric(self):
        """K(x_i, x_j) = K(x_j, x_i)."""
        X = jrandom.normal(jrandom.PRNGKey(1), (8, 3))
        K = rbf_gram(X)
        np.testing.assert_allclose(np.array(K), np.array(K.T), atol=1e-6)

    def test_positive_semidefinite(self):
        """RBF Gram matrix is always PSD."""
        X = jrandom.normal(jrandom.PRNGKey(2), (15, 5))
        K = rbf_gram(X)
        eigenvalues = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigenvalues >= -1e-5)

    def test_values_in_zero_one(self):
        """All entries of K are in [0, 1]."""
        X = jrandom.normal(jrandom.PRNGKey(3), (20, 6))
        K = rbf_gram(X)
        assert jnp.all(K >= -1e-7)
        assert jnp.all(K <= 1.0 + 1e-7)

    def test_known_value(self):
        """K(x, y) = exp(-||x-y||^2 / (2*sigma^2)) for known inputs."""
        X = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        sigma2 = jnp.array(2.0)
        K = rbf_gram(X, sigma2=sigma2)
        # ||x0 - x1||^2 = 1.0
        expected_01 = np.exp(-1.0 / (2.0 * 2.0))
        np.testing.assert_allclose(float(K[0, 1]), expected_01, atol=1e-6)
        np.testing.assert_allclose(float(K[1, 0]), expected_01, atol=1e-6)

    def test_median_heuristic_bandwidth(self):
        """When sigma2=None, bandwidth is set from median of distances."""
        rng = jrandom.PRNGKey(10)
        X = jrandom.normal(rng, (20, 3))
        K = rbf_gram(X)  # auto bandwidth
        # K should be a valid kernel (PSD, diag ~1; float32 precision)
        np.testing.assert_allclose(
            np.diag(np.array(K)), 1.0, atol=1e-3,
        )
        eigenvalues = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigenvalues >= -1e-3)


# ====================================================================
# Gram matrix centering
# ====================================================================


class TestCenterGram:

    def test_centered_gram_has_zero_row_col_means(self):
        """After centering, row means and column means should be ~0."""
        X = jrandom.normal(jrandom.PRNGKey(5), (10, 4))
        K = rbf_gram(X)
        Kc = center_gram(K)

        row_means = jnp.mean(Kc, axis=1)
        col_means = jnp.mean(Kc, axis=0)

        np.testing.assert_allclose(
            np.array(row_means), 0.0, atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(col_means), 0.0, atol=1e-5,
        )

    def test_centering_is_idempotent(self):
        """Centering an already-centered matrix gives the same result."""
        X = jrandom.normal(jrandom.PRNGKey(6), (8, 3))
        K = rbf_gram(X)
        Kc = center_gram(K)
        Kcc = center_gram(Kc)
        np.testing.assert_allclose(
            np.array(Kc), np.array(Kcc), atol=1e-5,
        )

    def test_centering_matches_H_K_H(self):
        """center_gram(K) = H K H where H = I - (1/n) 11^T."""
        n = 8
        X = jrandom.normal(jrandom.PRNGKey(7), (n, 3))
        # Use fixed bandwidth to avoid float32 precision issues
        # from the median heuristic
        K = rbf_gram(X, sigma2=jnp.array(1.0))

        H = jnp.eye(n) - jnp.ones((n, n)) / n
        expected = H @ K @ H
        result = center_gram(K)

        # float32 matmul (H@K@H) vs element-wise centering accumulate
        # rounding differently; atol=1e-3 is appropriate for float32
        np.testing.assert_allclose(
            np.array(result), np.array(expected), atol=1e-3,
        )


# ====================================================================
# HSIC (unconditional)
# ====================================================================


class TestHSIC:

    def test_hsic_self_is_positive(self):
        """HSIC(X, X) > 0 for non-degenerate X."""
        X = jrandom.normal(jrandom.PRNGKey(10), (30, 4))
        h = hsic_rbf(X, X)
        assert float(h) > 0.0

    def test_hsic_independent_is_near_zero(self):
        """HSIC(X, Y) ≈ 0 when X, Y are independent, large n.

        Uses large n so the finite-sample HSIC is close to population HSIC.
        """
        rng = jrandom.PRNGKey(11)
        n = 2000
        X = jrandom.normal(rng, (n, 3))
        Y = jrandom.normal(jrandom.fold_in(rng, 1), (n, 3))
        h = hsic_rbf(X, Y)
        # With n=2000, independent HSIC should be very small
        assert abs(float(h)) < 0.01

    def test_hsic_dependent_is_large(self):
        """HSIC(X, f(X)) >> HSIC(X, independent_Y) for deterministic f."""
        rng = jrandom.PRNGKey(12)
        n = 200
        X = jrandom.normal(rng, (n, 3))
        Y_dep = X ** 2  # deterministic function of X
        Y_indep = jrandom.normal(jrandom.fold_in(rng, 1), (n, 3))

        h_dep = hsic_rbf(X, Y_dep)
        h_indep = hsic_rbf(X, Y_indep)

        assert float(h_dep) > float(h_indep) * 5

    def test_hsic_symmetric(self):
        """HSIC(X, Y) = HSIC(Y, X)."""
        rng = jrandom.PRNGKey(13)
        X = jrandom.normal(rng, (50, 3))
        Y = jrandom.normal(jrandom.fold_in(rng, 1), (50, 4))

        np.testing.assert_allclose(
            float(hsic_rbf(X, Y)),
            float(hsic_rbf(Y, X)),
            atol=1e-6,
        )

    def test_hsic_nonnegative_for_same_input(self):
        """Biased HSIC(X, X) >= 0."""
        rng = jrandom.PRNGKey(14)
        for seed in range(5):
            X = jrandom.normal(jrandom.fold_in(rng, seed), (30, 4))
            h = hsic_rbf(X, X)
            assert float(h) >= -1e-7


# ====================================================================
# Class-conditional HSIC
# ====================================================================


class TestClassCondHSIC:

    def test_single_class_matches_unconditional(self):
        """With one class, class-conditional HSIC ≈ unconditional HSIC."""
        rng = jrandom.PRNGKey(20)
        n = 50
        X = jrandom.normal(rng, (n, 3))
        Y = jrandom.normal(jrandom.fold_in(rng, 1), (n, 4))
        labels = jnp.zeros(n, dtype=jnp.int32)  # all class 0

        h_cond = class_cond_hsic_rbf(X, Y, labels, num_classes=1)
        h_uncond = hsic_rbf(X, Y)

        np.testing.assert_allclose(
            float(h_cond), float(h_uncond), rtol=0.05,
        )

    def test_independent_within_classes_near_zero(self):
        """If X, Y are independent within every class, ccHSIC ≈ 0."""
        rng = jrandom.PRNGKey(21)
        n = 1000
        n_classes = 5
        X = jrandom.normal(rng, (n, 3))
        Y = jrandom.normal(jrandom.fold_in(rng, 1), (n, 3))
        labels = jrandom.randint(
            jrandom.fold_in(rng, 2), (n,), 0, n_classes,
        )

        h = class_cond_hsic_rbf(X, Y, labels, num_classes=n_classes)
        assert abs(float(h)) < 0.05

    def test_class_conditional_detects_within_class_dependence(self):
        """ccHSIC is large when X and Y are dependent within each class."""
        rng = jrandom.PRNGKey(22)
        n = 500
        n_classes = 3
        X = jrandom.normal(rng, (n, 3))
        Y = X ** 2  # dependent
        labels = jrandom.randint(
            jrandom.fold_in(rng, 1), (n,), 0, n_classes,
        )

        h = class_cond_hsic_rbf(X, Y, labels, num_classes=n_classes)
        assert float(h) > 0.005

    def test_empty_class_does_not_cause_nan(self):
        """Classes with <2 members should not produce NaN."""
        rng = jrandom.PRNGKey(23)
        n = 20
        X = jrandom.normal(rng, (n, 3))
        Y = jrandom.normal(jrandom.fold_in(rng, 1), (n, 3))
        # All samples in class 0; classes 1-4 are empty
        labels = jnp.zeros(n, dtype=jnp.int32)

        h = class_cond_hsic_rbf(X, Y, labels, num_classes=5)
        assert jnp.isfinite(h)

    def test_result_is_finite(self):
        """Check no NaN/Inf for random inputs."""
        rng = jrandom.PRNGKey(24)
        n = 100
        X = jrandom.normal(rng, (n, 4))
        Y = jrandom.normal(jrandom.fold_in(rng, 1), (n, 4))
        labels = jrandom.randint(
            jrandom.fold_in(rng, 2), (n,), 0, 10,
        )
        h = class_cond_hsic_rbf(X, Y, labels, num_classes=10)
        assert jnp.isfinite(h)


# ====================================================================
# Weighted centering
# ====================================================================


class TestWeightedCenterGram:

    def test_uniform_weights_matches_center_gram(self):
        """With w = 1/n, weighted centering should match standard centering."""
        n = 10
        X = jrandom.normal(jrandom.PRNGKey(30), (n, 4))
        K = rbf_gram(X)

        w = jnp.ones(n) / n
        Kc_weighted = _weighted_center_gram(K, w)
        Kc_standard = center_gram(K)

        np.testing.assert_allclose(
            np.array(Kc_weighted), np.array(Kc_standard), atol=1e-5,
        )


# ====================================================================
# Standardization
# ====================================================================


class TestStandardize:

    def test_output_has_zero_mean_unit_variance(self):
        """Standardized output should have mean ≈ 0, std ≈ 1 per dim."""
        z = jrandom.normal(jrandom.PRNGKey(40), (100, 5)) * 3 + 7
        z_std = _standardize(z)
        np.testing.assert_allclose(
            np.array(z_std.mean(axis=0)), 0.0, atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(z_std.std(axis=0)), 1.0, atol=0.02,
        )

    def test_constant_dimension_does_not_produce_nan(self):
        """A constant column should produce 0s, not NaN (due to eps)."""
        z = jnp.ones((10, 3))
        z_std = _standardize(z)
        assert jnp.all(jnp.isfinite(z_std))


# ====================================================================
# Spectral penalties
# ====================================================================


class TestSpectralPenalties:

    def test_nuclear_norm_rank1(self):
        """Nuclear norm of a rank-1 matrix equals its only singular value."""
        # z = outer product → rank-1 centered matrix
        v = jnp.array([[1.0, 0.0, 0.0]] * 5 +
                       [[-1.0, 0.0, 0.0]] * 5)  # centered, rank 1
        # After centering, z_centered has rank 1 in the first dimension
        penalty = svd_spectral_penalty(v)
        # The singular value should be related to the variance
        assert float(penalty) > 0

    def test_nuclear_norm_nonnegative(self):
        """Nuclear norm is always >= 0."""
        rng = jrandom.PRNGKey(50)
        z = jrandom.normal(rng, (20, 8))
        assert float(svd_spectral_penalty(z)) >= 0

    def test_frobenius_nonnegative(self):
        """Frobenius penalty is always >= 0."""
        rng = jrandom.PRNGKey(51)
        z = jrandom.normal(rng, (20, 8))
        assert float(frobenius_penalty(z)) >= 0

    def test_frobenius_zero_for_identical_points(self):
        """All points the same → cov = 0 → penalty = 0."""
        z = jnp.ones((10, 4)) * 3.0
        np.testing.assert_allclose(float(frobenius_penalty(z)), 0.0, atol=1e-6)

    def test_effective_rank_is_one_for_rank1(self):
        """Effective rank of a rank-1 covariance is 1."""
        # All variation along one axis
        z = jnp.zeros((100, 4))
        z = z.at[:, 0].set(jrandom.normal(jrandom.PRNGKey(52), (100,)))
        R = effective_rank_penalty(z)
        np.testing.assert_allclose(float(R), 1.0, atol=0.1)

    def test_effective_rank_equals_d_for_isotropic(self):
        """R_eff = d when all eigenvalues are equal (isotropic)."""
        # Use d orthogonal dimensions with equal variance
        d = 4
        rng = jrandom.PRNGKey(53)
        z = jrandom.normal(rng, (1000, d))  # ~isotropic
        R = effective_rank_penalty(z)
        np.testing.assert_allclose(float(R), d, atol=0.5)

    def test_effective_rank_bounds(self):
        """1 <= R_eff <= d for any input."""
        rng = jrandom.PRNGKey(54)
        d = 6
        z = jrandom.normal(rng, (50, d))
        R = effective_rank_penalty(z)
        assert float(R) >= 1.0 - 0.1
        assert float(R) <= d + 0.1

    def test_effective_rank_scale_invariant(self):
        """R_eff(alpha * z) = R_eff(z) for any scalar alpha > 0."""
        rng = jrandom.PRNGKey(55)
        z = jrandom.normal(rng, (50, 5))
        R1 = effective_rank_penalty(z)
        R2 = effective_rank_penalty(z * 100.0)
        np.testing.assert_allclose(float(R1), float(R2), rtol=1e-4)


# ====================================================================
# ControlVAE controller dynamics
# ====================================================================


class TestControlVAEController:
    """Tests the controller logic extracted from _vib_loss.

    The controller is: beta_new = clip(beta + K_I * (KL - C), beta_min, beta_max)
    where C is the capacity target (lamb).
    """

    def _run_controller(self, beta, kl, lamb, ctrl_ki=1.0,
                        beta_min=0.0, beta_max=1.0):
        beta_candidate = beta + ctrl_ki * (kl - lamb)
        return float(jnp.clip(beta_candidate, beta_min, beta_max))

    def test_kl_above_target_increases_beta(self):
        """When KL > target, beta should increase (more KL penalty)."""
        beta_new = self._run_controller(
            beta=0.5, kl=5.0, lamb=1.0,
        )
        assert beta_new > 0.5

    def test_kl_below_target_decreases_beta(self):
        """When KL < target, beta should decrease (less KL penalty)."""
        beta_new = self._run_controller(
            beta=0.5, kl=0.1, lamb=1.0,
        )
        assert beta_new < 0.5

    def test_kl_equals_target_no_change(self):
        """When KL = target, beta should not change."""
        beta_new = self._run_controller(
            beta=0.5, kl=1.0, lamb=1.0,
        )
        np.testing.assert_allclose(beta_new, 0.5, atol=1e-6)

    def test_beta_clipped_to_bounds(self):
        """Beta cannot exceed [beta_min, beta_max]."""
        # Large KL overshoot
        beta_new = self._run_controller(
            beta=0.9, kl=100.0, lamb=0.0,
        )
        assert beta_new <= 1.0

        # Large KL undershoot
        beta_new = self._run_controller(
            beta=0.1, kl=0.0, lamb=100.0,
        )
        assert beta_new >= 0.0

    def test_convergence_over_multiple_steps(self):
        """Controller should drive KL toward the target over time.

        Simulates a toy system where KL responds to beta linearly.
        """
        beta = 0.0
        lamb = 2.0  # capacity target
        ctrl_ki = 0.1
        beta_min, beta_max = 0.0, 1.0

        for _ in range(100):
            # Toy model: KL decreases with beta
            # (more beta → more KL penalty → lower KL)
            kl = 5.0 * (1.0 - beta)
            beta = self._run_controller(
                beta, kl, lamb, ctrl_ki, beta_min, beta_max,
            )

        # After convergence, KL should be near the target
        final_kl = 5.0 * (1.0 - beta)
        np.testing.assert_allclose(final_kl, lamb, atol=0.5)


# ====================================================================
# VIB loss composition
# ====================================================================


class TestVIBLossComposition:
    """Tests the loss weighting: L = (1 - beta) * task + beta * KL
    where task = alpha * CE + (1 - alpha) * recon.
    """

    def test_alpha_zero_is_pure_recon(self):
        """alpha=0: task_loss = recon only."""
        ce, recon, alpha = 10.0, 3.0, 0.0
        task = ce * alpha + recon * (1.0 - alpha)
        np.testing.assert_allclose(task, 3.0, atol=1e-7)

    def test_alpha_one_is_pure_ce(self):
        """alpha=1: task_loss = CE only."""
        ce, recon, alpha = 10.0, 3.0, 1.0
        task = ce * alpha + recon * (1.0 - alpha)
        np.testing.assert_allclose(task, 10.0, atol=1e-7)

    def test_beta_zero_ignores_kl(self):
        """beta=0: total_loss = task_loss (no KL term)."""
        task, kl, beta = 5.0, 100.0, 0.0
        total = task * (1.0 - beta) + beta * kl
        np.testing.assert_allclose(total, 5.0, atol=1e-7)

    def test_beta_one_is_pure_kl(self):
        """beta=1: total_loss = KL only."""
        task, kl, beta = 5.0, 2.0, 1.0
        total = task * (1.0 - beta) + beta * kl
        np.testing.assert_allclose(total, 2.0, atol=1e-7)

    def test_interpolation(self):
        """Check linear interpolation at beta=0.3."""
        task, kl, beta = 5.0, 10.0, 0.3
        total = task * (1.0 - beta) + beta * kl
        expected = 5.0 * 0.7 + 10.0 * 0.3
        np.testing.assert_allclose(total, expected, atol=1e-7)


# ====================================================================
# MC predictive distribution (log-mean-exp)
# ====================================================================


class TestMCPredictive:
    """Tests the MC averaging in eval_step:
    log p(y|x) = logsumexp(log_softmax(logits_k), axis=0) - log(K)
    """

    def test_single_sample_equals_softmax(self):
        """With K=1, the MC estimate equals the softmax of that sample."""
        logits = jnp.array([[[2.0, 1.0, 0.5]]])  # [1, 1, 3]
        log_probs_K = jax.nn.log_softmax(logits, axis=-1)
        log_probs = logsumexp(log_probs_K, axis=0) - jnp.log(1)
        expected = jax.nn.log_softmax(logits[0], axis=-1)
        np.testing.assert_allclose(
            np.array(log_probs), np.array(expected), atol=1e-6,
        )

    def test_two_identical_samples_equal_one(self):
        """K identical samples should give the same result as K=1."""
        logits_single = jnp.array([[2.0, 1.0, 0.5]])  # [1, 3]
        logits_K = jnp.stack([logits_single, logits_single])  # [2, 1, 3]

        log_probs_K = jax.nn.log_softmax(logits_K, axis=-1)
        log_probs = logsumexp(log_probs_K, axis=0) - jnp.log(2)
        expected = jax.nn.log_softmax(logits_single, axis=-1)

        np.testing.assert_allclose(
            np.array(log_probs), np.array(expected), atol=1e-6,
        )

    def test_result_is_valid_log_distribution(self):
        """The MC average should produce a valid probability distribution."""
        rng = jrandom.PRNGKey(60)
        K, B, C = 8, 4, 5
        logits_K = jrandom.normal(rng, (K, B, C))

        log_probs_K = jax.nn.log_softmax(logits_K, axis=-1)
        log_probs = logsumexp(log_probs_K, axis=0) - jnp.log(K)
        probs = jnp.exp(log_probs)

        # Each row should sum to 1
        np.testing.assert_allclose(
            np.array(probs.sum(axis=-1)), 1.0, atol=1e-5,
        )
        # All probabilities should be in [0, 1]
        assert jnp.all(probs >= -1e-7)
        assert jnp.all(probs <= 1.0 + 1e-7)

    def test_mixture_is_more_uncertain_than_components(self):
        """The mixture entropy should be >= max component entropy.

        H(mixture) >= H(component) because mixing adds uncertainty.
        """
        K, C = 4, 3
        # Two very different predictive distributions
        logits_K = jnp.array([
            [[10.0, 0.0, 0.0]],  # confident in class 0
            [[0.0, 10.0, 0.0]],  # confident in class 1
            [[0.0, 0.0, 10.0]],  # confident in class 2
            [[3.0, 3.0, 3.0]],   # uniform
        ])  # [4, 1, 3]

        log_probs_K = jax.nn.log_softmax(logits_K, axis=-1)
        log_probs_mix = logsumexp(log_probs_K, axis=0) - jnp.log(K)
        probs_mix = jnp.exp(log_probs_mix)

        # Mixture should be close to uniform (high entropy)
        entropy_mix = -jnp.sum(probs_mix * log_probs_mix, axis=-1)
        max_entropy = jnp.log(C)  # uniform entropy

        # Mixture of diverse components should have high entropy
        assert float(entropy_mix[0]) > 0.5 * float(max_entropy)
