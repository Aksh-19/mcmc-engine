"""Target distributions for MCMC sampling.

This module provides the base `Distribution` class and concrete implementations
of common probability distributions. All distributions work in log-space to
avoid numerical underflow.

Planned distributions:
    - Normal (1D and multivariate)
    - Beta
    - Gamma
    - Banana (Rosenbrock) — for testing sampler robustness
    - Custom user-defined distributions
"""
