"""Tests for MCMC diagnostics."""

import numpy as np
from mcmc.diagnostics import effective_sample_size, r_hat


def test_effective_sample_size_independent():
    """An independent chain (white noise) should have ESS ≈ N."""
    rng = np.random.default_rng(42)
    N = 10_000
    chain = rng.normal(0, 1, size=N)
    
    ess = effective_sample_size(chain)
    
    # It won't be exactly 10,000 due to random statistical noise, but very close
    assert 9000 < ess < 11000


def test_effective_sample_size_correlated():
    """A highly correlated chain should have ESS << N."""
    rng = np.random.default_rng(42)
    N = 10_000
    chain = np.zeros(N)
    
    # Create an AR(1) process with high correlation (0.9)
    chain[0] = rng.normal(0, 1)
    for i in range(1, N):
        chain[i] = 0.9 * chain[i-1] + rng.normal(0, 0.1)
        
    ess = effective_sample_size(chain)
    
    # Highly correlated means it's worth far fewer independent samples
    assert ess < 2000


def test_r_hat_converged():
    """If multiple chains sample the exact same distribution, R-hat ≈ 1.0."""
    rng = np.random.default_rng(42)
    N = 1000
    
    # 4 chains, all sampling standard normal
    chain1 = rng.normal(0, 1, size=N)
    chain2 = rng.normal(0, 1, size=N)
    chain3 = rng.normal(0, 1, size=N)
    chain4 = rng.normal(0, 1, size=N)
    
    chains = np.array([chain1, chain2, chain3, chain4])
    
    r = r_hat(chains)
    assert abs(r - 1.0) < 0.05


def test_r_hat_not_converged():
    """If chains are stuck in different modes, R-hat > 1.0."""
    rng = np.random.default_rng(42)
    N = 1000
    
    # 2 chains sampling entirely different areas (stuck!)
    chain1 = rng.normal(0, 1, size=N)     # centered at 0
    chain2 = rng.normal(10, 1, size=N)    # centered at 10
    
    chains = np.array([chain1, chain2])
    
    r = r_hat(chains)
    # Between-variance is massive compared to within-variance
    assert r > 2.0
