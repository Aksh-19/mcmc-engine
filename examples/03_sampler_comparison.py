"""
Phase 2 — Sampler Showdown: MH vs Gibbs vs Slice

This script runs all three samplers on a highly correlated 2D Gaussian.
Highly correlated distributions are notoriously difficult for standard MH.

Usage:
    python examples/03_sampler_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmc.samplers import MetropolisHastings, GibbsSampler, SliceSampler

# We will sample from a 2D Gaussian with correlation rho = 0.95
# This creates a narrow, diagonal "ridge" of probability.
RHO = 0.95

# 1. Target for MH and Slice (requires log_prob)
class CorrelatedGaussian:
    def log_prob(self, x):
        # -0.5 * (x1^2 - 2*rho*x1*x2 + x2^2) / (1 - rho^2)
        z = x[0]**2 - 2*RHO*x[0]*x[1] + x[1]**2
        return -0.5 * z / (1 - RHO**2)

target = CorrelatedGaussian()

# 2. Target for Gibbs (requires full conditionals)
std_cond = np.sqrt(1.0 - RHO**2)
def cond_x0(x, rng): return rng.normal(RHO * x[1], std_cond)
def cond_x1(x, rng): return rng.normal(RHO * x[0], std_cond)

# --- RUN THE SAMPLERS ---
num_samples = 2000
initial = [-3.0, 3.0]  # Start far from the mean (0,0)

print("Running Metropolis-Hastings...")
mh = MetropolisHastings(target, proposal_scale=0.5, seed=42)
chain_mh = mh.run(num_samples, initial)

print("\nRunning Gibbs...")
gibbs = GibbsSampler([cond_x0, cond_x1], seed=42)
chain_gibbs = gibbs.run(num_samples, initial)

print("\nRunning Slice...")
slice_samp = SliceSampler(target, w=1.0, seed=42)
chain_slice = slice_samp.run(num_samples, initial)

# --- PLOTTING ---
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
samplers = [("Metropolis-Hastings", chain_mh, "red"), 
            ("Gibbs Sampling", chain_gibbs, "green"), 
            ("Slice Sampling", chain_slice, "blue")]

for i, (name, chain, color) in enumerate(samplers):
    # Scatter plot (Path)
    ax_scatter = axes[i, 0]
    ax_scatter.plot(chain[:200, 0], chain[:200, 1], alpha=0.6, color=color, marker='.', markersize=4, lw=1)
    ax_scatter.set_title(f"{name} (First 200 steps)")
    ax_scatter.set_xlim(-4, 4)
    ax_scatter.set_ylim(-4, 4)
    
    # Trace plot (x0 over time)
    ax_trace = axes[i, 1]
    ax_trace.plot(chain[:, 0], color=color, lw=0.5)
    ax_trace.set_title(f"{name} Trace (x0)")
    ax_trace.set_ylim(-4, 4)

plt.tight_layout()
plt.savefig("examples/plots/03_sampler_comparison.png", dpi=150)
print("\nSaved comparison plot to: examples/plots/03_sampler_comparison.png")
