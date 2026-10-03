"""Tests for the Gibbs sampler."""

import numpy as np
from mcmc.samplers import GibbsSampler


def test_gibbs_2d_gaussian():
    """
    Test Gibbs sampling on a 2D Gaussian with correlation.
    True distribution:
        mu_1 = 0, mu_2 = 0
        var_1 = 1, var_2 = 1
        correlation rho = 0.8
    """
    rho = 0.8
    std_cond = np.sqrt(1.0 - rho**2)

    # Full conditional for x0 given x1
    def cond_x0(x, rng):
        return rng.normal(rho * x[1], std_cond)

    # Full conditional for x1 given x0
    def cond_x1(x, rng):
        return rng.normal(rho * x[0], std_cond)

    sampler = GibbsSampler(full_conditionals=[cond_x0, cond_x1], seed=42)
    chain = sampler.run(num_samples=20_000, initial_state=[5.0, -5.0])

    # Discard burn-in
    samples = chain[2000:]
    
    # 1. Means should be ~0
    assert abs(np.mean(samples[:, 0])) < 0.05
    assert abs(np.mean(samples[:, 1])) < 0.05
    
    # 2. Variances should be ~1
    assert abs(np.var(samples[:, 0]) - 1.0) < 0.1
    assert abs(np.var(samples[:, 1]) - 1.0) < 0.1
    
    # 3. Covariance should be ~0.8 (the off-diagonal element)
    cov_matrix = np.cov(samples, rowvar=False)
    assert abs(cov_matrix[0, 1] - 0.8) < 0.1

def test_gibbs_linear_regression():
    """
    Test Gibbs sampling on Bayesian Linear Regression.
    Model: y = a*x + b + noise
    True parameters: a = 3.0, b = 2.0
    """
    rng = np.random.default_rng(42)
    N_pts = 100
    x_data = rng.uniform(-5, 5, N_pts)
    true_a, true_b = 3.0, 2.0
    noise_var = 1.0
    y_data = true_a * x_data + true_b + rng.normal(0, np.sqrt(noise_var), N_pts)

    prior_var = 100.0

    # Conditional for 'a' (slope) given 'b'
    def cond_a(state, local_rng):
        b = state[1]
        residuals = y_data - b
        sum_x_sq = np.sum(x_data**2)
        post_var = 1.0 / (sum_x_sq / noise_var + 1.0 / prior_var)
        post_mean = post_var * (np.sum(x_data * residuals) / noise_var)
        return local_rng.normal(post_mean, np.sqrt(post_var))

    # Conditional for 'b' (intercept) given 'a'
    def cond_b(state, local_rng):
        a = state[0]
        residuals = y_data - a * x_data
        post_var = 1.0 / (N_pts / noise_var + 1.0 / prior_var)
        post_mean = post_var * (np.sum(residuals) / noise_var)
        return local_rng.normal(post_mean, np.sqrt(post_var))

    # state = [a, b]
    sampler = GibbsSampler(full_conditionals=[cond_a, cond_b], seed=42)
    chain = sampler.run(num_samples=5000, initial_state=[0.0, 0.0])

    samples = chain[1000:]
    
    # We should recover the true slope (3.0) and intercept (2.0)
    assert abs(np.mean(samples[:, 0]) - true_a) < 0.1
    assert abs(np.mean(samples[:, 1]) - true_b) < 0.1
