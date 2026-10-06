"""Tests for the No-U-Turn Sampler (NUTS)."""

import numpy as np
from mcmc.distributions import Normal, Banana
from mcmc.samplers import NUTSSampler

def test_nuts_1d_normal():
    """NUTS should sample 1D Normal without needing a num_steps parameter."""
    target = Normal(mu=5.0, sigma=2.0)
    
    # Notice: No num_steps provided! Only step_size and max_depth.
    sampler = NUTSSampler(target, step_size=0.5, max_depth=5, seed=42)
    chain = sampler.run(num_samples=1000, initial_state=[0.0])
    
    samples = chain[200:]  # Discard burn-in
    
    assert abs(np.mean(samples) - 5.0) < 0.2
    assert abs(np.std(samples) - 2.0) < 0.2

def test_nuts_2d_banana():
    """NUTS should handle the curved Banana distribution efficiently."""
    target = Banana(a=1.0, b=10.0)
    
    sampler = NUTSSampler(target, step_size=0.05, max_depth=7, seed=42)
    chain = sampler.run(num_samples=1000, initial_state=[0.0, 0.0])
    
    samples = chain[300:]
    
    assert abs(np.mean(samples[:, 0]) - 1.0) < 0.3
