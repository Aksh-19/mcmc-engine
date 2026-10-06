"""
Phase 7 — Capstone 2: Bayesian Change-Point Detection

Detects the exact day a stealthy cyber-attack began by finding a hidden 
shift in noisy server traffic. Uses a "sigmoid relaxation" trick to make 
the discrete day-shift differentiable so NUTS can solve it.

Usage:
    python examples/07_change_point.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mcmc.samplers import NUTSSampler

class CyberAttackModel:
    def __init__(self, data):
        self.data = data
        self.days = len(data)

    def log_prob(self, params):
        # 4 Parameters:
        # mu1: Normal traffic baseline
        # mu2: Breached traffic baseline
        # tau: The day the attack started
        # log_sigma: The natural noise/variance of the server
        mu1, mu2, tau, log_sigma = params[0], params[1], params[2], params[3]
        sigma = log_sigma.exp()
        
        # 1. Priors (Keep guesses reasonable)
        # We assume traffic is roughly around 500
        log_p = -0.5 * ((mu1 - 500.0) / 100.0)**2
        log_p = log_p - 0.5 * ((mu2 - 500.0) / 100.0)**2
        
        # The attack could have happened any day between 0 and 100
        # A weak prior centered at day 50
        log_p = log_p - 0.5 * ((tau - 50.0) / 50.0)**2
        
        # 2. Likelihood (The Server Logs)
        for t in range(self.days):
            traffic = self.data[t]
            
            # The Sigmoid Relaxation Trick!
            # We multiply by 0.3 (a "temperature" parameter) to stretch the S-curve out.
            # If it's too steep, the gradient vanishes and the skateboarder gets stuck!
            time_diff = t - tau
            weight = 1.0 / (1.0 + (-0.3 * time_diff).exp())
            
            expected_traffic = mu1 * (1.0 - weight) + mu2 * weight
            
            log_p = log_p - log_sigma - 0.5 * ((traffic - expected_traffic) / sigma)**2
            
        return log_p

# ==========================================
# 1. GENERATE STEALTH HACK DATA
# ==========================================
print("Simulating 100 days of noisy server logs...")
days = 100
true_mu1 = 500.0   # Normal daily traffic (Terabytes)
true_mu2 = 700.0   # Hacker steals a massive 200 TB a day!
true_tau = 72.0    # The hacker activates the malware on Day 72
true_sigma = 35.0  # Natural server variance

rng = np.random.default_rng(42)
traffic_data = np.zeros(days)

for t in range(days):
    if t < true_tau:
        traffic_data[t] = rng.normal(true_mu1, true_sigma)
    else:
        traffic_data[t] = rng.normal(true_mu2, true_sigma)

# ==========================================
# 2. RUN BAYESIAN INFERENCE WITH NUTS
# ==========================================
print("\nRunning NUTS to find the exact day of the breach...")
model = CyberAttackModel(traffic_data)

# Initial guess: 
# We MUST guess a different mu1 and mu2. If mu1 == mu2, the gradient for tau 
# mathematically multiplies by zero, and the physics engine gets completely stuck!
initial_guess = [500.0, 600.0, 50.0, 3.5]

sampler = NUTSSampler(model, step_size=0.03, max_depth=6, seed=42)
chain = sampler.run(num_samples=400, initial_state=initial_guess)

samples = chain[100:]  # Discard burn-in

# ==========================================
# 3. RESULTS & PLOTTING
# ==========================================
pred_mu1 = np.mean(samples[:, 0])
pred_mu2 = np.mean(samples[:, 1])
pred_tau = np.mean(samples[:, 2])

print("\n--- CYBER-ATTACK DETECTED ---")
print(f"True Breach Day:      Day {true_tau}")
print(f"Predicted Breach Day: Day {pred_tau:.1f}")
print(f"Data Stolen:          {pred_mu2 - pred_mu1:.1f} TB per day")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the raw, noisy server data
ax.scatter(np.arange(days), traffic_data, color='gray', alpha=0.6, label='Raw Server Traffic')
ax.plot(np.arange(days), traffic_data, color='gray', alpha=0.2)

# Plot the true hidden timeline
ax.plot([0, true_tau], [true_mu1, true_mu1], color='cyan', lw=3, label='True Normal Baseline')
ax.plot([true_tau, days], [true_mu2, true_mu2], color='red', lw=3, label='True Breached Baseline')

# Overlay the Bayesian Posterior Samples (Uncertainty)
for i in range(0, len(samples), 5):
    s_mu1 = samples[i, 0]
    s_mu2 = samples[i, 1]
    s_tau = samples[i, 2]
    
    # We plot the sigmoid curves the math generated
    t = np.linspace(0, days, 200)
    curve = s_mu1 * (1.0 - 1.0/(1.0 + np.exp(-(t - s_tau)))) + s_mu2 * (1.0/(1.0 + np.exp(-(t - s_tau))))
    ax.plot(t, curve, color='purple', alpha=0.05)

# Highlight the predicted change point
ax.axvline(pred_tau, color='black', linestyle='--', lw=2, label=f'Predicted Breach: Day {pred_tau:.1f}')

ax.set_title("Stealth Cyber-Attack: Bayesian Change-Point Detection", fontsize=14, fontweight='bold')
ax.set_xlabel("Days")
ax.set_ylabel("Server Traffic (TB)")
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig("examples/plots/07_change_point.png", dpi=150)
print("\nSaved timeline plot to: examples/plots/07_change_point.png")
