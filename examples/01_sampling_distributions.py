"""
Phase 1 — Sampling from Normal, Beta, and Banana distributions.

Run this script to generate histograms comparing MH samples against true distributions.

Usage:
    python examples/01_sampling_distributions.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmc.distributions import Normal, Beta, Banana
from mcmc.samplers import MetropolisHastings

# ============================================================
# 1. Sample from Normal(0, 1)
# ============================================================
print("=" * 50)
print("Sampling from Normal(0, 1)")
print("=" * 50)

target = Normal(mu=0, sigma=1)
sampler = MetropolisHastings(target, proposal_scale=1.0, seed=42)
chain = sampler.run(num_samples=50_000, initial_state=np.array([0.0]))

samples = chain[5000:]  # discard burn-in

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(samples[:, 0], bins=80, density=True, alpha=0.7, label="MH samples")
x_grid = np.linspace(-4, 4, 200)
ax.plot(x_grid, np.exp(-0.5 * x_grid**2) / np.sqrt(2 * np.pi), "r-", lw=2, label="True Normal(0,1)")
ax.set_title("Metropolis-Hastings: Normal(0, 1)")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig("examples/plots/01_normal_histogram.png", dpi=150)
print("Saved: examples/plots/01_normal_histogram.png")

# ============================================================
# 2. Sample from Beta(2, 5)
# ============================================================
print("\n" + "=" * 50)
print("Sampling from Beta(2, 5)")
print("=" * 50)

target = Beta(alpha=2, beta=5)
sampler = MetropolisHastings(target, proposal_scale=0.1, seed=42)
chain = sampler.run(num_samples=50_000, initial_state=np.array([0.5]))

samples = chain[5000:]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(samples[:, 0], bins=80, density=True, alpha=0.7, label="MH samples")

# True Beta PDF (computed manually, no scipy)
x_grid = np.linspace(0.001, 0.999, 200)
alpha, beta_param = 2, 5
# unnormalized beta, then normalize by numerical integration
log_pdf = (alpha - 1) * np.log(x_grid) + (beta_param - 1) * np.log(1 - x_grid)
pdf = np.exp(log_pdf)
pdf = pdf / np.trapezoid(pdf, x_grid)  # normalize
ax.plot(x_grid, pdf, "r-", lw=2, label="True Beta(2,5)")
ax.set_title("Metropolis-Hastings: Beta(2, 5)")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig("examples/plots/01_beta_histogram.png", dpi=150)
print("Saved: examples/plots/01_beta_histogram.png")

# ============================================================
# 3. Sample from Banana (2D)
# ============================================================
print("\n" + "=" * 50)
print("Sampling from Banana (2D)")
print("=" * 50)

target = Banana(a=1.0, b=10.0)
sampler = MetropolisHastings(target, proposal_scale=0.1, seed=42)
chain = sampler.run(num_samples=100_000, initial_state=np.array([0.0, 0.0]))

samples = chain[10000:]

fig, ax = plt.subplots(1, 1, figsize=(8, 7))
ax.scatter(samples[::5, 0], samples[::5, 1], s=1, alpha=0.3, c="steelblue")
ax.set_title("Metropolis-Hastings: Banana Distribution (2D)")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("examples/plots/01_banana_scatter.png", dpi=150)
print("Saved: examples/plots/01_banana_scatter.png")

print("\nDone! Check examples/plots/ for the figures.")
