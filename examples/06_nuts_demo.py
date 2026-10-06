"""
Phase 6 — NUTS Demo: The Power of No Tuning

This script races a poorly tuned HMC against NUTS.
Notice how HMC gets stuck doing a slow random walk because the user guessed 
a bad trajectory length. NUTS figures out the perfect trajectory length 
automatically and explores the entire Banana in the same number of samples!

Usage:
    python examples/06_nuts_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmc.distributions import Banana
from mcmc.samplers import HMCSampler, NUTSSampler

target = Banana(a=1.0, b=10.0)
num_samples = 400
initial = [0.0, 3.0]

# 1. HMC (Badly Tuned - user guessed num_steps too low)
print("Running HMC (Badly tuned: num_steps=3)...")
hmc = HMCSampler(target, step_size=0.05, num_steps=3, seed=42)
chain_hmc = hmc.run(num_samples, initial)

# 2. NUTS (No tuning required for trajectory length!)
print("\nRunning NUTS (Auto-tuning trajectory length)...")
nuts = NUTSSampler(target, step_size=0.05, max_depth=10, seed=42)
chain_nuts = nuts.run(num_samples, initial)

# --- PLOTTING ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(chain_hmc[:, 0], chain_hmc[:, 1], color='orange', alpha=0.6, marker='.', lw=0.5)
axes[0].set_title("HMC (Badly tuned L: stuck in random walk)")
axes[0].set_xlim(-3, 5)
axes[0].set_ylim(-2, 16)

axes[1].plot(chain_nuts[:, 0], chain_nuts[:, 1], color='purple', alpha=0.6, marker='.', lw=0.5)
axes[1].set_title("NUTS (Auto-tuned: Explores everything!)")
axes[1].set_xlim(-3, 5)
axes[1].set_ylim(-2, 16)

plt.tight_layout()
plt.savefig("examples/plots/06_nuts_demo.png", dpi=150)
print("\nSaved comparison plot to: examples/plots/06_nuts_demo.png")
