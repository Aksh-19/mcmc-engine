"""
Phase 1 — Effect of proposal scale on Metropolis-Hastings.

Demonstrates what happens when σ is too small, just right, or too large.

Usage:
    python examples/02_proposal_tuning.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmc.distributions import Normal
from mcmc.samplers import MetropolisHastings

target = Normal(mu=0, sigma=1)
scales = [0.01, 1.0, 100.0]
labels = ["σ=0.01 (too small)", "σ=1.0 (just right)", "σ=100 (too large)"]
num_samples = 10_000

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for row, (scale, label) in enumerate(zip(scales, labels)):
    sampler = MetropolisHastings(target, proposal_scale=scale, seed=42)
    chain = sampler.run(num_samples=num_samples, initial_state=np.array([0.0]))

    # Trace plot (left column): sample value vs iteration
    axes[row, 0].plot(chain[:2000, 0], lw=0.5, color="steelblue")
    axes[row, 0].set_title(f"Trace plot — {label}")
    axes[row, 0].set_xlabel("Iteration")
    axes[row, 0].set_ylabel("x")
    axes[row, 0].set_ylim(-5, 5)

    # Histogram (right column): compare to true distribution
    axes[row, 1].hist(chain[1000:, 0], bins=60, density=True, alpha=0.7, label="MH samples")
    x_grid = np.linspace(-4, 4, 200)
    axes[row, 1].plot(x_grid, np.exp(-0.5 * x_grid**2) / np.sqrt(2 * np.pi), "r-", lw=2, label="True")
    axes[row, 1].set_title(f"Histogram — {label}")
    axes[row, 1].set_xlabel("x")
    axes[row, 1].legend()

plt.tight_layout()
plt.savefig("examples/plots/02_proposal_tuning.png", dpi=150)
print("Saved: examples/plots/02_proposal_tuning.png")
print("\nNotice:")
print("  - σ=0.01: trace plot shows tiny steps, histogram looks OK but mixing is slow")
print("  - σ=1.0:  trace plot shows good exploration, histogram matches perfectly")
print("  - σ=100:  trace plot shows the chain getting 'stuck' (flat lines), histogram is spiky")
