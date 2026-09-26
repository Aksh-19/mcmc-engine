"""MCMC sampling algorithms.

This module provides implementations of various MCMC samplers, from the
foundational Metropolis-Hastings to the state-of-the-art NUTS.

Planned samplers:
    - MetropolisHastings — random-walk and independent variants (Phase 1)
    - GibbsSampler — coordinate-wise sampling with full conditionals (Phase 2)
    - SliceSampler — adaptive, tuning-free sampling (Phase 2)
    - HMCSampler — Hamiltonian Monte Carlo with leapfrog integration (Phase 5)
    - NUTSSampler — No-U-Turn Sampler with auto-tuning (Phase 6)
"""
