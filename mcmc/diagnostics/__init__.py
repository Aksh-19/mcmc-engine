"""Convergence diagnostics for MCMC chains.

This module provides tools to assess whether your MCMC chains have converged
and how many effective independent samples you have.

Planned diagnostics:
    - trace_plot() — visual check: is the chain exploring or stuck?
    - autocorrelation() — how many steps until samples are independent?
    - effective_sample_size() — how many independent samples do you really have?
    - r_hat() — Gelman-Rubin: do multiple chains agree? (THE most important diagnostic)
    - summary() — table of mean, std, ESS, R-hat per parameter
"""
