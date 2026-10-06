"""
Phase 7 — Capstone App: Bayesian Geiger Counter 3D

Finds a hidden radioactive source in a 3D room using noisy Geiger counter 
measurements, Poisson statistics, and the No-U-Turn Sampler (NUTS).

Usage:
    python examples/07_geiger_counter.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmc.samplers import NUTSSampler

class GeigerModel:
    def __init__(self, drone_coords, counts, background=5.0):
        self.drone_coords = drone_coords
        self.counts = counts
        self.background = background

    def log_prob(self, params):
        # params = [x, y, z, log_I0]
        # We parameterize intensity in log-space so it can never be negative.
        # This stabilizes the physics simulation and prevents math domain errors!
        xs, ys, zs, log_I0 = params[0], params[1], params[2], params[3]
        I0 = log_I0.exp()
        
        # 1. Priors (keep the sampler from wandering to infinity)
        log_p = -0.5 * ((xs - 5.0)**2 + (ys - 5.0)**2 + (zs - 5.0)**2) / 25.0
        # Weak prior on log_I0 (guessing log(1000) ≈ 6.9)
        log_p = log_p - 0.5 * ((log_I0 - 6.0) ** 2) / 4.0
        
        # 2. Likelihood (The Physics: Inverse Square Law + Poisson noise)
        for i in range(len(self.counts)):
            xi, yi, zi = self.drone_coords[i]
            ci = self.counts[i]
            
            # Distance squared to the hidden source
            dist_sq = (xi - xs)**2 + (yi - ys)**2 + (zi - zs)**2
            
            # Expected radiation at the drone's location
            lam = (I0 / (dist_sq + 1.0)) + self.background
            
            # Poisson log-likelihood: c * log(lambda) - lambda
            log_p = log_p + (ci * lam.log() - lam)
            
        return log_p

# ==========================================
# 1. GENERATE SYNTHETIC REAL-WORLD DATA
# ==========================================
print("Simulating dense drone spiral scan through warehouse...")
true_x, true_y, true_z = 7.5, 2.0, 8.0  # The hidden source!
true_I0 = 1200.0                        # Very radioactive
background = 10.0

# Drone flies in a structured sweeping spiral taking 200 measurements
t = np.linspace(0, 6 * np.pi, 200)
drone_x = 5 + 4.5 * np.cos(t)
drone_y = 5 + 4.5 * np.sin(t)
drone_z = np.linspace(1, 9, 200)
drone_coords = np.column_stack((drone_x, drone_y, drone_z))

rng = np.random.default_rng(42)
counts = []
for coords in drone_coords:
    dist_sq = np.sum((coords - [true_x, true_y, true_z])**2)
    expected_radiation = true_I0 / (dist_sq + 1.0) + background
    counts.append(rng.poisson(expected_radiation))
    
counts = np.array(counts)
print(f"Max radiation detected: {np.max(counts)} CPM")

# ==========================================
# 2. RUN BAYESIAN INFERENCE WITH NUTS
# ==========================================
print("\nRunning NUTS to find the source... (This does 4D autodiff physics, give it a few seconds)")
model = GeigerModel(drone_coords, counts, background=background)

# We start with a generic guess: middle of the room, log(100) ≈ 4.6
initial_guess = [5.0, 5.0, 5.0, 4.6]

# Lower step size because 200 data points creates steep gradients
sampler = NUTSSampler(model, step_size=0.01, max_depth=6, seed=42)
# 300 samples is plenty for NUTS because it mixes so well
chain = sampler.run(num_samples=300, initial_state=initial_guess)

samples = chain[100:]  # Discard burn-in

# ==========================================
# 3. RESULTS & PLOTTING
# ==========================================
pred_x = np.mean(samples[:, 0])
pred_y = np.mean(samples[:, 1])
pred_z = np.mean(samples[:, 2])
pred_I0 = np.exp(np.mean(samples[:, 3]))

print("\n--- RESULTS ---")
print(f"True Location:      X={true_x:.1f}, Y={true_y:.1f}, Z={true_z:.1f} (I0={true_I0})")
print(f"Predicted Location: X={pred_x:.1f}, Y={pred_y:.1f}, Z={pred_z:.1f} (I0={pred_I0:.0f})")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot drone path
scatter = ax.scatter(drone_coords[:, 0], drone_coords[:, 1], drone_coords[:, 2], 
                     c=counts, cmap='YlOrRd', s=50, label='Drone Measurements')
plt.colorbar(scatter, label='Radiation (CPM)', shrink=0.5)

# Plot True Source
ax.scatter(true_x, true_y, true_z, color='blue', s=200, marker='*', label='True Source')

# Plot Predicted Posterior (a cloud of where the math thinks it is)
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color='green', alpha=0.1, s=10, label='MCMC Posterior')

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
ax.set_title("Bayesian Geiger Counter 3D")
ax.legend()

plt.tight_layout()
plt.savefig("examples/plots/07_geiger_counter.png", dpi=150)
print("\nSaved 3D plot to: examples/plots/07_geiger_counter.png")
