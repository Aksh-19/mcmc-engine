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
# 1. GENERATE DATA (Dense Sensor Grid)
# ==========================================
print("Simulating a noisy grid of sensors in a 6x6 room...")
true_x, true_y, true_z = 4.0, 4.0, 1.0  # Source elevated slightly at (4,4)
true_I0 = 150.0  # Weaker source
background = 40.0 # High background noise

# 8x8 uniform grid of sensors on the floor (64 total points)
x_grid = np.linspace(0, 6, 8)
y_grid = np.linspace(0, 6, 8)
xv, yv = np.meshgrid(x_grid, y_grid)
drone_coords = np.column_stack((xv.ravel(), yv.ravel(), np.zeros_like(xv.ravel())))

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
# Start guess in center of the 6x6 room
chain = sampler.run(num_samples=300, initial_state=[3.0, 3.0, 1.0, 4.0])[100:]

# ==========================================
# 3. SEABORN CONTINUOUS HEATMAP
# ==========================================
print("\nGenerating Seaborn Continuous Heatmap...")
sns.set_theme(style="darkgrid")

g = sns.JointGrid(x=chain[:, 0], y=chain[:, 1], height=9, space=0.1)

# Plot a glowing continuous KDE map for the posterior probability
g.plot_joint(sns.kdeplot, fill=True, cmap="magma", thresh=0.01, levels=25)

# Marginals showing the 1D probabilities
g.plot_marginals(sns.kdeplot, color="purple", fill=True, alpha=0.7)

# Overlay the RAW SENSOR READINGS as a background grid!
sensor_scatter = g.ax_joint.scatter(drone_coords[:, 0], drone_coords[:, 1], c=counts, cmap='magma', 
                   s=400, marker='s', alpha=0.3, label="Raw Sensor Readings")

# Add a colorbar for the raw sensor readings
cbar = plt.colorbar(sensor_scatter, ax=g.ax_joint, pad=0.1, fraction=0.05)
cbar.set_label("Raw Sensor Readings (CPM)", fontsize=10)

# Overlay True Source location
g.ax_joint.scatter(true_x, true_y, color='cyan', marker='*', s=300, edgecolor='black', label="True Source (X,Y)")

g.ax_joint.set_xlim(0, 6)
g.ax_joint.set_ylim(0, 6)
g.ax_joint.set_xlabel("Room X Coordinate", fontsize=12)
g.ax_joint.set_ylabel("Room Y Coordinate", fontsize=12)
g.ax_joint.legend(loc="upper left")

g.fig.suptitle("Bayesian Radiation Map (NUTS Posterior Density)", y=1.03, fontsize=16, fontweight='bold')

plt.savefig("examples/plots/07_geiger_heatmap.png", dpi=150, bbox_inches='tight')
print("Saved glowing map to: examples/plots/07_geiger_heatmap.png")
