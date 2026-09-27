"""Tests for probability distributions."""

import numpy as np
import pytest
from mcmc.distributions import Normal, Beta, Banana


class TestNormal:
    """Tests for the Normal distribution."""

    def test_log_prob_at_mean(self):
        """log_prob is highest at the mean."""
        dist = Normal(mu=0, sigma=1)
        assert dist.log_prob(0.0) > dist.log_prob(1.0)
        assert dist.log_prob(0.0) > dist.log_prob(-1.0)

    def test_log_prob_known_value(self):
        """log_prob(0) for standard Normal should be -0.5*log(2π) ≈ -0.9189."""
        dist = Normal(mu=0, sigma=1)
        expected = -0.5 * np.log(2 * np.pi)
        assert abs(dist.log_prob(0.0) - expected) < 1e-10

    def test_symmetry(self):
        """Normal(0,1) is symmetric: log_prob(x) == log_prob(-x)."""
        dist = Normal(mu=0, sigma=1)
        assert abs(dist.log_prob(2.0) - dist.log_prob(-2.0)) < 1e-10

    def test_different_mu_sigma(self):
        """log_prob is highest at mu for any mu, sigma."""
        dist = Normal(mu=5, sigma=2)
        assert dist.log_prob(5.0) > dist.log_prob(3.0)
        assert dist.log_prob(5.0) > dist.log_prob(7.0)

    def test_callable(self):
        """dist(x) should work the same as dist.log_prob(x)."""
        dist = Normal(mu=0, sigma=1)
        assert dist(1.5) == dist.log_prob(1.5)


class TestBeta:
    """Tests for the Beta distribution."""

    def test_valid_range(self):
        """log_prob returns finite values for x in (0, 1)."""
        dist = Beta(alpha=2, beta=5)
        result = dist.log_prob(0.3)
        assert np.isfinite(result)

    def test_out_of_range_low(self):
        """log_prob returns -inf for x <= 0."""
        dist = Beta(alpha=2, beta=5)
        assert dist.log_prob(0.0) == -np.inf
        assert dist.log_prob(-0.5) == -np.inf

    def test_out_of_range_high(self):
        """log_prob returns -inf for x >= 1."""
        dist = Beta(alpha=2, beta=5)
        assert dist.log_prob(1.0) == -np.inf
        assert dist.log_prob(1.5) == -np.inf

    def test_mode_location(self):
        """Beta(2,5) has mode at (2-1)/(2+5-2) = 0.2. log_prob should be highest near there."""
        dist = Beta(alpha=2, beta=5)
        assert dist.log_prob(0.2) > dist.log_prob(0.01)
        assert dist.log_prob(0.2) > dist.log_prob(0.9)

    def test_known_value(self):
        """Beta(2,5).log_prob(0.3) should equal (2-1)*log(0.3) + (5-1)*log(0.7)."""
        dist = Beta(alpha=2, beta=5)
        expected = (2 - 1) * np.log(0.3) + (5 - 1) * np.log(0.7)
        assert abs(dist.log_prob(0.3) - expected) < 1e-10


class TestBanana:
    """Tests for the Banana (Rosenbrock) distribution."""

    def test_peak_at_a_a_squared(self):
        """Banana(a=1, b=10) has its peak at [1, 1] where log_prob = 0."""
        dist = Banana(a=1.0, b=10.0)
        assert dist.log_prob(np.array([1.0, 1.0])) == 0.0

    def test_away_from_peak(self):
        """Points away from the peak have lower log_prob."""
        dist = Banana(a=1.0, b=10.0)
        peak_val = dist.log_prob(np.array([1.0, 1.0]))
        other_val = dist.log_prob(np.array([0.0, 0.0]))
        assert peak_val > other_val

    def test_callable(self):
        """dist(x) should work the same as dist.log_prob(x)."""
        dist = Banana()
        x = np.array([0.5, 0.5])
        assert dist(x) == dist.log_prob(x)
