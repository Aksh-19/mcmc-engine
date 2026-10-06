# Capstone 1: Bayesian Geiger Counter 3D

## The Problem Statement
Imagine a drone flying through a massive warehouse, or a person walking around a room with a cheap Geiger counter. The radiation counts are highly noisy, probabilistic, and fluctuate wildly. How do you find the exact 3D coordinates of a hidden radioactive source using only a few noisy data points?

This capstone applies the custom-built NUTS (No-U-Turn Sampler) engine to solve this real-world physics problem. By feeding the engine a time-series of 3D coordinates $(X, Y, Z)$ and the raw Geiger counter CPM (Counts Per Minute), the engine dynamically calculates a glowing 3D "probability heatmap," pinpointing the exact coordinates and intensity of the hidden source.

## The Physics & The Math

### 1. The Inverse-Square Law
Radiation drops off quadratically based on distance. If you double your distance from the source, the radiation drops to a quarter. The expected radiation at any point is:
$$ \lambda = \frac{I_0}{d^2 + 1} + B $$
Where $I_0$ is the source intensity, $d$ is the distance to the source, and $B$ is the natural background radiation of the room.

### 2. Poisson Noise
Geiger counters don't measure exact continuous values; they measure discrete radioactive particle impacts. These impacts follow a **Poisson distribution**. The MCMC engine evaluates the likelihood of the drone's actual clicks against the expected $\lambda$. 

### 3. Posterior Collapse (The Background Noise)
If there is no background noise ($B=0$), finding the source is mathematically trivial. However, real rooms have natural background radiation. When the drone is far from the source, the weak signal is completely drowned out by the background noise. This creates a highly complex, flat probability space that standard samplers (like Metropolis-Hastings) get completely lost in.

## Engineering Challenges

### The Negative Radiation Crash
Physics-based samplers like HMC and NUTS explore space by rolling down mathematical slopes. Sometimes, momentum pushes a parameter into negative territory. If the sampler guesses the radiation intensity $I_0 = -50$, the Poisson math (`log(-50)`) crashes the entire simulation.

**The Solution: Log-Parameterization**
To fix this, we parameterize the engine using `log_I0`. The sampler is free to guess *any* number from $-\infty$ to $+\infty$. Before doing the physics math, we convert it back via $I_0 = e^{\log I_0}$. Exponentials are strictly positive, completely stabilizing the physics simulation!

## Advanced Application: 8-Dimensional NUTS
The capstone was expanded to map **two** hidden sources simultaneously. The engine had to solve an 8-dimensional posterior $(X_1, Y_1, Z_1, I_1, X_2, Y_2, Z_2, I_2)$. Because a drone only picks up a single blended radiation signal at any given point, the NUTS engine had to mathematically untangle the overlapping radiation clouds to perfectly locate both distinct sources in 3D space.
