"""
Phase 3 — Diagnostics Demo: Why R-hat is Critical

This script creates a bimodal distribution (two separate mountains of probability).
We run two chains starting in different mountains.
Each trace plot looks fine individually, but R-hat warns us they haven't converged!

Usage:
    python examples/04_diagnostics_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmc.samplers import MetropolisHastings
from mcmc.diagnostics import effective_sample_size, r_hat, trace_plot

# 1. A Bimodal Target Distribution (Two Mountains far apart)
class Bimodal:
    def log_prob(self, x):
        # Two modes at -5 and +5.
        # It's so far apart that a random walk sampler will get stuck in one.
        val = x[0]
        prob1 = np.exp(-0.5 * (val - 5)**2)
        prob2 = np.exp(-0.5 * (val + 5)**2)
        return np.log(prob1 + prob2 + 1e-10)

target = Bimodal()

# 2. Run two chains from completely different starting points
print("Running Chain 1 (Starts at +5)...")
sampler1 = MetropolisHastings(target, proposal_scale=0.5, seed=42)
chain1 = sampler1.run(num_samples=5000, initial_state=[5.0])

print("Running Chain 2 (Starts at -5)...")
sampler2 = MetropolisHastings(target, proposal_scale=0.5, seed=43)
chain2 = sampler2.run(num_samples=5000, initial_state=[-5.0])

# Remove burn-in (first 1000)
c1 = chain1[1000:, 0]
c2 = chain2[1000:, 0]

# 3. Calculate Diagnostics
print("\n--- Diagnostics ---")
ess1 = effective_sample_size(c1)
ess2 = effective_sample_size(c2)
print(f"Chain 1 ESS: {ess1:.1f} / 4000")
print(f"Chain 2 ESS: {ess2:.1f} / 4000")

# Calculate R-hat (Needs shape: M chains x N samples)
chains_matrix = np.array([c1, c2])
r = r_hat(chains_matrix)
print(f"\nR-hat (Gelman-Rubin): {r:.3f}")
print("Rule of thumb: > 1.05 means BAD convergence.")
print("Even though ESS is high, R-hat caught that the chains are stuck in different modes!")

# 4. Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].plot(c1, color='blue', alpha=0.7, label='Chain 1')
axes[0].plot(c2, color='red', alpha=0.7, label='Chain 2')
axes[0].set_title(f"Trace Plots overlayed (R-hat = {r:.2f})")
axes[0].legend()

# Histogram
axes[1].hist(c1, bins=30, color='blue', alpha=0.5, density=True)
axes[1].hist(c2, bins=30, color='red', alpha=0.5, density=True)
axes[1].set_title("Histograms: They think the true distribution is entirely different!")

plt.tight_layout()
plt.savefig("examples/plots/04_diagnostics_demo.png", dpi=150)
print("\nSaved plot to: examples/plots/04_diagnostics_demo.png")
