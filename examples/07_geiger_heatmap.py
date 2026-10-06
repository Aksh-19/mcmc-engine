"""
Phase 7 — Capstone App: 2D Continuous Radiation Heatmap

This script takes the 3D Bayesian Geiger counter problem and visualizes 
the posterior probability as a continuous, glowing 2D heatmap (X vs Y) 
using Seaborn KDE plots. 

Usage:
    python examples/07_geiger_heatmap.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mcmc.samplers import NUTSSampler

class GeigerModel:
    def __init__(self, drone_coords, counts, background=10.0):
        self.drone_coords = drone_coords
        self.counts = counts
        self.background = background

    def log_prob(self, params):
        xs, ys, zs, log_I0 = params[0], params[1], params[2], params[3]
        I0 = log_I0.exp()
        
        log_p = -0.5 * ((xs - 5.0)**2 + (ys - 5.0)**2 + (zs - 5.0)**2) / 25.0
        log_p = log_p - 0.5 * ((log_I0 - 6.0) ** 2) / 4.0
        
        for i in range(len(self.counts)):
            xi, yi, zi = self.drone_coords[i]
            ci = self.counts[i]
            dist_sq = (xi - xs)**2 + (yi - ys)**2 + (zi - zs)**2
            lam = (I0 / (dist_sq + 1.0)) + self.background
            log_p = log_p + (ci * lam.log() - lam)
            
        return log_p

# ==========================================
# 1. GENERATE DATA
# ==========================================
print("Simulating dense drone spiral scan through warehouse...")
true_x, true_y, true_z = 7.5, 2.0, 8.0
true_I0 = 1200.0
background = 10.0

t = np.linspace(0, 6 * np.pi, 200)
drone_x = 5 + 4.5 * np.cos(t)
drone_y = 5 + 4.5 * np.sin(t)
drone_z = np.linspace(1, 9, 200)
drone_coords = np.column_stack((drone_x, drone_y, drone_z))

rng = np.random.default_rng(42)
counts = []
for coords in drone_coords:
    dist_sq = np.sum((coords - [true_x, true_y, true_z])**2)
    counts.append(rng.poisson((true_I0 / (dist_sq + 1.0)) + background))
counts = np.array(counts)

# ==========================================
# 2. RUN BAYESIAN INFERENCE WITH NUTS
# ==========================================
print("\nRunning NUTS to find the source...")
model = GeigerModel(drone_coords, counts, background=background)
sampler = NUTSSampler(model, step_size=0.01, max_depth=6, seed=42)
chain = sampler.run(num_samples=300, initial_state=[5.0, 5.0, 5.0, 4.6])[100:]

# ==========================================
# 3. SEABORN CONTINUOUS HEATMAP
# ==========================================
print("\nGenerating Seaborn Continuous Heatmap...")
sns.set_theme(style="darkgrid")

# We plot the X and Y coordinates of the posterior chain
g = sns.JointGrid(x=chain[:, 0], y=chain[:, 1], height=9, space=0.1)

# Plot a glowing continuous KDE map using the "magma" colormap
g.plot_joint(sns.kdeplot, fill=True, cmap="magma", thresh=0.01, levels=20)

# Marginals showing the exact bell curves for X and Y
g.plot_marginals(sns.kdeplot, color="purple", fill=True, alpha=0.7)

# Overlay True Source location as a cyan star
g.ax_joint.scatter(true_x, true_y, color='cyan', marker='*', s=400, edgecolor='black', label="True Source (X,Y)")

# Overlay the drone spiral faintly in the background
g.ax_joint.plot(drone_x, drone_y, color='gray', alpha=0.4, linestyle='--', label="Drone Path")
# Put tiny dots on the drone path colored by radiation intensity
g.ax_joint.scatter(drone_x, drone_y, c=counts, cmap='magma', s=15, alpha=0.7, edgecolor='none')

g.ax_joint.set_xlim(0, 10)
g.ax_joint.set_ylim(0, 10)
g.ax_joint.set_xlabel("Warehouse X Coordinate", fontsize=12)
g.ax_joint.set_ylabel("Warehouse Y Coordinate", fontsize=12)
g.ax_joint.legend(loc="upper left")

g.fig.suptitle("Bayesian Radiation Map (Continuous Posterior KDE)", y=1.03, fontsize=16, fontweight='bold')

plt.savefig("examples/plots/07_geiger_heatmap.png", dpi=150, bbox_inches='tight')
print("Saved glowing map to: examples/plots/07_geiger_heatmap.png")
