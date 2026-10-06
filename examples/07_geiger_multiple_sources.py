"""
Phase 7 — Advanced Capstone: Multi-Source Geiger Counter 3D

Finds TWO hidden radioactive sources in a large 30x30x10 warehouse using 
a dense drone flight path, Poisson statistics, and an 8-Dimensional NUTS model!

Usage:
    python examples/07_geiger_multiple_sources.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmc.samplers import NUTSSampler

class MultiGeigerModel:
    def __init__(self, drone_coords, counts, background=10.0):
        self.drone_coords = drone_coords
        self.counts = counts
        self.background = background

    def log_prob(self, params):
        # We now have 8 Parameters! 
        # Source 1: (x1, y1, z1, log_I1)
        # Source 2: (x2, y2, z2, log_I2)
        x1, y1, z1, log_I1 = params[0], params[1], params[2], params[3]
        x2, y2, z2, log_I2 = params[4], params[5], params[6], params[7]
        
        I1 = log_I1.exp()
        I2 = log_I2.exp()
        
        # 1. Priors
        # We use weak priors to softly suggest Source 1 is on the left side of the room
        # and Source 2 is on the right. This prevents "Label Switching" (where the 
        # sampler swaps their identities back and forth, confusing the math).
        log_p = -0.5 * ((x1 - 10.0)**2 + (y1 - 15.0)**2 + (z1 - 5.0)**2) / 400.0
        log_p = log_p - 0.5 * ((x2 - 20.0)**2 + (y2 - 15.0)**2 + (z2 - 5.0)**2) / 400.0
        
        # Weak prior on intensity (guessing log(1000) ≈ 6.9)
        log_p = log_p - 0.5 * ((log_I1 - 6.0)**2 + (log_I2 - 6.0)**2) / 10.0
        
        # 2. Likelihood (Inverse Square Law + Poisson noise)
        for i in range(len(self.counts)):
            xi, yi, zi = self.drone_coords[i]
            ci = self.counts[i]
            
            dist1_sq = (xi - x1)**2 + (yi - y1)**2 + (zi - z1)**2
            dist2_sq = (xi - x2)**2 + (yi - y2)**2 + (zi - z2)**2
            
            # The radiation at the drone is the SUM of both sources!
            lam = (I1 / (dist1_sq + 1.0)) + (I2 / (dist2_sq + 1.0)) + self.background
            
            log_p = log_p + (ci * lam.log() - lam)
            
        return log_p

# ==========================================
# 1. GENERATE ADVANCED SYNTHETIC DATA
# ==========================================
print("Simulating drone spiral scan through the warehouse...")
# Source 1 (Low and left)
true_x1, true_y1, true_z1 = 8.0, 10.0, 2.0  
true_I1 = 1500.0

# Source 2 (High and right)
true_x2, true_y2, true_z2 = 22.0, 25.0, 8.0  
true_I2 = 2500.0                        

background = 10.0

# Create a dense spiral flight path for the drone
t = np.linspace(0, 8 * np.pi, 250)
drone_x = 15 + 13 * np.cos(t)
drone_y = 15 + 13 * np.sin(t)
drone_z = np.linspace(1, 9, 250)
drone_coords = np.column_stack((drone_x, drone_y, drone_z))

rng = np.random.default_rng(42)
counts = []
for coords in drone_coords:
    d1_sq = np.sum((coords - [true_x1, true_y1, true_z1])**2)
    d2_sq = np.sum((coords - [true_x2, true_y2, true_z2])**2)
    
    expected_radiation = (true_I1 / (d1_sq + 1.0)) + (true_I2 / (d2_sq + 1.0)) + background
    counts.append(rng.poisson(expected_radiation))
    
counts = np.array(counts)
print(f"Max radiation detected: {np.max(counts)} CPM")

# ==========================================
# 2. RUN BAYESIAN INFERENCE WITH NUTS (8D)
# ==========================================
print("\nRunning 8-Dimensional NUTS to map both sources... (This will take ~10-20 seconds)")
model = MultiGeigerModel(drone_coords, counts, background=background)

# Initial guess
# Source 1 guess: x=10, y=15, z=5, log_I=4.6 (100)
# Source 2 guess: x=20, y=15, z=5, log_I=4.6 (100)
initial_guess = [10.0, 15.0, 5.0, 4.6, 20.0, 15.0, 5.0, 4.6]

# Lower step size significantly because 250 data points creates very steep gradients!
sampler = NUTSSampler(model, step_size=0.005, max_depth=5, seed=42)
chain = sampler.run(num_samples=300, initial_state=initial_guess)

samples = chain[100:]  # Discard burn-in

# ==========================================
# 3. RESULTS & PLOTTING
# ==========================================
pred_x1, pred_y1, pred_z1 = np.mean(samples[:, 0]), np.mean(samples[:, 1]), np.mean(samples[:, 2])
pred_I1 = np.exp(np.mean(samples[:, 3]))

pred_x2, pred_y2, pred_z2 = np.mean(samples[:, 4]), np.mean(samples[:, 5]), np.mean(samples[:, 6])
pred_I2 = np.exp(np.mean(samples[:, 7]))

print("\n--- RESULTS ---")
print("SOURCE 1:")
print(f"  True:      X={true_x1:.1f}, Y={true_y1:.1f}, Z={true_z1:.1f} (I0={true_I1})")
print(f"  Predicted: X={pred_x1:.1f}, Y={pred_y1:.1f}, Z={pred_z1:.1f} (I0={pred_I1:.0f})")
print("\nSOURCE 2:")
print(f"  True:      X={true_x2:.1f}, Y={true_y2:.1f}, Z={true_z2:.1f} (I0={true_I2})")
print(f"  Predicted: X={pred_x2:.1f}, Y={pred_y2:.1f}, Z={pred_z2:.1f} (I0={pred_I2:.0f})")

# Advanced Plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot drone flight path
scatter = ax.scatter(drone_coords[:, 0], drone_coords[:, 1], drone_coords[:, 2], 
                     c=counts, cmap='magma', s=25, label='Drone Flight Path')
plt.colorbar(scatter, label='Radiation (CPM)', shrink=0.5, pad=0.1)

# Plot True Sources
ax.scatter(true_x1, true_y1, true_z1, color='cyan', s=300, marker='*', edgecolor='black', label='True Source 1')
ax.scatter(true_x2, true_y2, true_z2, color='lime', s=300, marker='*', edgecolor='black', label='True Source 2')

# Plot Predicted Posterior Clouds
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color='cyan', alpha=0.1, s=15, label='Posterior 1')
ax.scatter(samples[:, 4], samples[:, 5], samples[:, 6], color='lime', alpha=0.1, s=15, label='Posterior 2')

ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(0, 10)
ax.set_title("8D Bayesian Mapping: Drone Spiral Path & Multiple Sources")
ax.legend()

plt.tight_layout()
plt.savefig("examples/plots/07_geiger_advanced.png", dpi=150)
print("\nSaved advanced 3D plot to: examples/plots/07_geiger_advanced.png")
