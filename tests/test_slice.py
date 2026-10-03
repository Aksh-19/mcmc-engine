"""Tests for the Slice sampler."""

import numpy as np
from mcmc.distributions import Normal, Beta, Banana
from mcmc.samplers import SliceSampler


def test_slice_1d_normal():
    """Samples from Normal(5,2) should have mean ≈ 5, std ≈ 2."""
    target = Normal(mu=5.0, sigma=2.0)
    sampler = SliceSampler(target, w=1.0, seed=42)
    chain = sampler.run(num_samples=10_000, initial_state=0.0)
    
    samples = chain[1000:]  # discard burn-in
    
    assert abs(np.mean(samples) - 5.0) < 0.1
    assert abs(np.std(samples) - 2.0) < 0.1


def test_slice_1d_beta():
    """Samples from Beta(2,5) should have mean ≈ 2/7."""
    target = Beta(alpha=2, beta=5)
    sampler = SliceSampler(target, w=0.5, seed=42)
    chain = sampler.run(num_samples=10_000, initial_state=0.5)
    
    samples = chain[1000:]
    expected_mean = 2.0 / 7.0
    
    assert abs(np.mean(samples) - expected_mean) < 0.05


def test_slice_2d_banana():
    """Slice sampler should handle the 2D Banana distribution effortlessly."""
    target = Banana(a=1.0, b=10.0)
    sampler = SliceSampler(target, w=1.0, seed=42)
    chain = sampler.run(num_samples=10_000, initial_state=[0.0, 0.0])
    
    samples = chain[2000:]
    
    # Mean of x_0 for standard Banana should be close to 'a' (1.0)
    assert abs(np.mean(samples[:, 0]) - 1.0) < 0.2
