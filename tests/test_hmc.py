"""Tests for Hamiltonian Monte Carlo (HMC)."""

import numpy as np
from mcmc.distributions import Normal, Banana
from mcmc.samplers import HMCSampler

def test_hmc_1d_normal():
    """HMC should correctly sample from a 1D Normal distribution."""
    target = Normal(mu=5.0, sigma=2.0)
    
    # step_size * num_steps determines trajectory length.
    sampler = HMCSampler(target, step_size=0.5, num_steps=5, seed=42)
    chain = sampler.run(num_samples=2000, initial_state=[0.0])
    
    samples = chain[500:]  # discard burn-in
    
    assert abs(np.mean(samples) - 5.0) < 0.2
    assert abs(np.std(samples) - 2.0) < 0.2

def test_hmc_2d_banana():
    """HMC should smoothly navigate the curved 2D Banana distribution."""
    target = Banana(a=1.0, b=10.0)
    
    # Banana has high curvature, so we need a smaller step size for stability
    sampler = HMCSampler(target, step_size=0.05, num_steps=15, seed=42)
    chain = sampler.run(num_samples=2000, initial_state=[0.0, 0.0])
    
    samples = chain[500:]
    
    # The mean of x0 in this standard Banana distribution is roughly 'a' (1.0)
    assert abs(np.mean(samples[:, 0]) - 1.0) < 0.3
