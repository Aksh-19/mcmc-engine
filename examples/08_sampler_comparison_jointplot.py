"""
Bonus — Sampler Accuracy Comparison (Seaborn JointGrid)

This script runs Metropolis-Hastings (MH), Hamiltonian Monte Carlo (HMC), 
and the No-U-Turn Sampler (NUTS) on the highly curved 2D Banana distribution. 

It overlays all three probability contours and marginal distributions 
using a beautiful Seaborn JointGrid to compare their exploration accuracy.

Usage:
    python examples/08_sampler_comparison_jointplot.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mcmc.distributions import Banana
from mcmc.samplers import MetropolisHastings, HMCSampler, NUTSSampler

# Set Seaborn aesthetic style
sns.set_theme(style="whitegrid")

target = Banana(a=1.0, b=10.0)
initial = [0.0, 3.0]
num_samples = 2000

# 1. Run MH
print("Running Metropolis-Hastings...")
mh = MetropolisHastings(target, proposal_scale=0.15, seed=42)
chain_mh = mh.run(num_samples, initial)[500:]

# 2. Run HMC (Poorly tuned to show typical user error)
print("\nRunning HMC...")
hmc = HMCSampler(target, step_size=0.05, num_steps=3, seed=42)
chain_hmc = hmc.run(num_samples, initial)[500:]

# 3. Run NUTS (Zero tuning)
print("\nRunning NUTS...")
nuts = NUTSSampler(target, step_size=0.05, max_depth=7, seed=42)
chain_nuts = nuts.run(num_samples, initial)[500:]

# ==========================================
# SEABORN JOINTGRID PLOTTING
# ==========================================
print("\nGenerating Seaborn JointGrid Plot...")
g = sns.JointGrid(height=9, ratio=4, space=0.1)

# Plot MH (Red)
sns.kdeplot(x=chain_mh[:, 0], y=chain_mh[:, 1], ax=g.ax_joint, color='red', alpha=0.5, levels=4)
sns.kdeplot(x=chain_mh[:, 0], ax=g.ax_marg_x, color='red', fill=True, alpha=0.3, label='Metropolis-Hastings')
sns.kdeplot(y=chain_mh[:, 1], ax=g.ax_marg_y, color='red', fill=True, alpha=0.3)

# Plot HMC (Blue)
sns.kdeplot(x=chain_hmc[:, 0], y=chain_hmc[:, 1], ax=g.ax_joint, color='blue', alpha=0.5, levels=4)
sns.kdeplot(x=chain_hmc[:, 0], ax=g.ax_marg_x, color='blue', fill=True, alpha=0.3, label='HMC (Poorly Tuned)')
sns.kdeplot(y=chain_hmc[:, 1], ax=g.ax_marg_y, color='blue', fill=True, alpha=0.3)

# Plot NUTS (Green)
sns.kdeplot(x=chain_nuts[:, 0], y=chain_nuts[:, 1], ax=g.ax_joint, color='green', alpha=0.9, levels=6, linewidths=2)
sns.kdeplot(x=chain_nuts[:, 0], ax=g.ax_marg_x, color='green', fill=True, alpha=0.6, label='NUTS (Perfect Auto-tuning)')
sns.kdeplot(y=chain_nuts[:, 1], ax=g.ax_marg_y, color='green', fill=True, alpha=0.6)

# Aesthetics
g.ax_joint.set_xlim(-4, 6)
g.ax_joint.set_ylim(-2, 16)
g.ax_joint.set_xlabel("Dimension 1 (Linear)", fontsize=12)
g.ax_joint.set_ylabel("Dimension 2 (Curved)", fontsize=12)

# Legend on the top marginal plot
g.ax_marg_x.legend(loc='upper right', bbox_to_anchor=(1.0, 1.2), frameon=True)

g.fig.suptitle("Sampler Accuracy Comparison (Seaborn KDE Overlays)", y=1.03, fontsize=16, fontweight='bold')

plt.savefig("examples/plots/08_sampler_comparison.png", dpi=150, bbox_inches='tight')
print("Saved beautiful JointGrid plot to: examples/plots/08_sampler_comparison.png")
