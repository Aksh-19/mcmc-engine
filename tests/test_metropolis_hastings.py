"""Tests for the Metropolis-Hastings sampler."""

import numpy as np
import pytest
from mcmc.distributions import Normal, Beta, Banana
from mcmc.samplers import MetropolisHastings


class TestMetropolisHastings1D:
    """Test MH on 1D distributions."""

    def test_normal_mean(self):
        """Samples from Normal(0,1) should have mean ≈ 0."""
        target = Normal(mu=0, sigma=1)
        sampler = MetropolisHastings(target, proposal_scale=1.0, seed=42)
        chain = sampler.run(num_samples=50_000, initial_state=np.array([0.0]))

        # Discard first 5000 as burn-in (chain needs time to "forget" the starting point)
        samples = chain[5000:]
        assert abs(np.mean(samples) - 0.0) < 0.05

    def test_normal_std(self):
        """Samples from Normal(0,1) should have std ≈ 1."""
        target = Normal(mu=0, sigma=1)
        sampler = MetropolisHastings(target, proposal_scale=1.0, seed=42)
        chain = sampler.run(num_samples=50_000, initial_state=np.array([0.0]))

        samples = chain[5000:]
        assert abs(np.std(samples) - 1.0) < 0.05

    def test_normal_different_mu(self):
        """Samples from Normal(5,2) should have mean ≈ 5."""
        target = Normal(mu=5, sigma=2)
        sampler = MetropolisHastings(target, proposal_scale=2.0, seed=42)
        chain = sampler.run(num_samples=50_000, initial_state=np.array([0.0]))

        samples = chain[5000:]
        assert abs(np.mean(samples) - 5.0) < 0.15

    def test_beta_mean(self):
        """Samples from Beta(2,5) should have mean ≈ 2/7 ≈ 0.286."""
        target = Beta(alpha=2, beta=5)
        sampler = MetropolisHastings(target, proposal_scale=0.1, seed=42)
        chain = sampler.run(num_samples=50_000, initial_state=np.array([0.5]))

        samples = chain[5000:]
        expected_mean = 2.0 / 7.0  # 0.2857
        assert abs(np.mean(samples) - expected_mean) < 0.05

    def test_acceptance_rate_reasonable(self, capsys):
        """Acceptance rate should be between 10% and 90% for well-tuned proposal."""
        target = Normal(mu=0, sigma=1)
        sampler = MetropolisHastings(target, proposal_scale=1.0, seed=42)
        sampler.run(num_samples=10_000, initial_state=np.array([0.0]))

        captured = capsys.readouterr()
        # Parse "Acceptance rate: 0.XXX" from printed output
        rate = float(captured.out.strip().split(": ")[1])
        assert 0.1 < rate < 0.9


class TestMetropolisHastings2D:
    """Test MH on 2D distributions."""

    def test_banana_peak(self):
        """Samples from Banana should cluster around the ridge y = x²."""
        target = Banana(a=1.0, b=10.0)
        sampler = MetropolisHastings(target, proposal_scale=0.1, seed=42)
        chain = sampler.run(num_samples=50_000, initial_state=np.array([0.0, 0.0]))

        samples = chain[10000:]  # longer burn-in for harder distribution
        mean_x = np.mean(samples[:, 0])
        # Mean of x should be near a=1.0
        assert abs(mean_x - 1.0) < 0.3

    def test_chain_shape(self):
        """Chain should have shape (num_samples, dim)."""
        target = Banana()
        sampler = MetropolisHastings(target, proposal_scale=0.1, seed=42)
        chain = sampler.run(num_samples=1000, initial_state=np.array([0.0, 0.0]))

        assert chain.shape == (1000, 2)
