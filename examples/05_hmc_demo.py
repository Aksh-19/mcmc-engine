"""
Phase 5 — HMC Demo: Physics in Action

This script compares Metropolis-Hastings (MH) against Hamiltonian Monte Carlo (HMC)
on a highly curved "Banana" distribution.
Notice how much faster HMC explores the extremes of the banana because it follows
the mathematical curves using gravity!

Usage:
    python examples/05_hmc_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmc.distributions import Banana
from mcmc.samplers import MetropolisHastings, HMCSampler

target = Banana(a=1.0, b=10.0)
num_samples = 1500
initial = [0.0, 3.0]

# 1. Run Metropolis-Hastings
print("Running Metropolis-Hastings (Blind guessing)...")
mh = MetropolisHastings(target, proposal_scale=0.15, seed=42)
chain_mh = mh.run(num_samples, initial)

# 2. Run HMC
print("\nRunning HMC (Physics simulation)...")
# step_size determines how fast the skater rolls, num_steps is how long they roll
hmc = HMCSampler(target, step_size=0.05, num_steps=10, seed=42)
chain_hmc = hmc.run(num_samples, initial)

# --- PLOTTING ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot MH
axes[0].plot(chain_mh[:, 0], chain_mh[:, 1], color='red', alpha=0.5, marker='.', lw=0.5)
axes[0].set_title("Metropolis-Hastings (Slow, stuck in middle)")
axes[0].set_xlim(-4, 6)
axes[0].set_ylim(-2, 16)

# Plot HMC
axes[1].plot(chain_hmc[:, 0], chain_hmc[:, 1], color='green', alpha=0.5, marker='.', lw=0.5)
axes[1].set_title("HMC (Smoothly explores the whole curve)")
axes[1].set_xlim(-4, 6)
axes[1].set_ylim(-2, 16)

plt.tight_layout()
plt.savefig("examples/plots/05_hmc_demo.png", dpi=150)
print("\nSaved comparison plot to: examples/plots/05_hmc_demo.png")
