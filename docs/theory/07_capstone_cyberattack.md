# Capstone 2: Stealth Cyber-Attack (Change-Point Detection)

## The Problem Statement
You are a cybersecurity analyst at a major tech company. Every single day, terabytes of data flow through your servers. The traffic is highly chaotic, swinging wildly up and down based on normal user activity. 

An Advanced Persistent Threat (APT) hacker has compromised a server. Instead of doing a massive, noisy data dump (which would trigger alarms), they deployed a "sleeper" malware. On a specific, unknown day, the malware woke up and started siphoning off a tiny, constant trickle of encrypted user data.

The stolen data volume is so small that it is completely hidden inside the massive daily spikes of normal server traffic. Standard monitoring tools see nothing. We must use Bayesian Change-Point Detection to find the exact day the breach began.

## The Model
We need to find three hidden parameters in the noise:
1. $\mu_1$: The normal daily traffic baseline.
2. $\mu_2$: The compromised traffic baseline (which tells us exactly how much data is being stolen).
3. $\tau$: The **Change-Point** — the exact day the hacker activated the malware.

## Engineering Challenges

### The "Cliff" of Discrete Variables
There is a massive mathematical problem: our NUTS sampler requires smooth, continuous physics (differentiable gradients) to roll its physics-based "skateboarder" around the probability space. 

But a cyber-attack activating on "Day 67" is a discrete integer. In math, a sudden ON/OFF switch creates a sheer cliff. There is no slope. If the physics engine hits this cliff, it cannot calculate a gradient, and the sampler crashes or gets hopelessly stuck.

### The Solution: Continuous Sigmoid Relaxation
To solve this, we use a brilliant Bayesian trick called **Continuous Relaxation**. 
Instead of modeling the attack as a harsh IF/ELSE switch, we model the shift as a microscopic, smooth S-curve using the `sigmoid` function. 

As the timeline crosses $\tau$, the traffic smoothly transitions from $\mu_1$ to $\mu_2$:
$$ \text{Expected Traffic}(t) = \mu_1 \times (1 - w) + \mu_2 \times w $$
Where $w = \sigma(k \cdot (t - \tau))$

Because it is a smooth curve, our Autodiff engine can calculate the slope at any point! The NUTS skateboarder can now mathematically roll down the timeline of the server logs and settle into the exact hour the hack began.

### The Vanishing Gradient
During testing, we discovered that if the S-curve is too steep, it creates flat plateaus on either side. If the engine's initial guess was too far away from the true attack day, the gradient was exactly 0.0 (the Vanishing Gradient problem). The physics engine felt zero gravity and failed to move.

By introducing a "temperature" parameter ($k=0.3$) to the sigmoid, we stretched the steep cliff out into a gentle ramp spanning across the timeline, guaranteeing the physics engine could "feel" the pull of the true attack day from anywhere on the calendar!
