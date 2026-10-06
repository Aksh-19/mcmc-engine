# Physics-Based Sampling: HMC & NUTS

In high dimensions (e.g., a 100-parameter machine learning model), the probability space is mostly empty, and the "good" parameters live on a tiny, narrow ridge. Random-walk samplers (like Metropolis-Hastings) will step off this ridge 99.9% of the time and get rejected. 

We need a sampler that *knows where the ridge is*.

---

## 1. Hamiltonian Monte Carlo (HMC)

HMC imagines your probability distribution as a massive, smooth concrete skateboard bowl.
*   **Potential Energy ($U$):** The height/depth of the bowl. We set this to $-\log(Probability)$. High probability = deep bowl.
*   **Kinetic Energy ($K$):** The speed of the skateboarder.
*   **Gravity:** The slope of the bowl (calculated by our Autodiff engine).

**The Algorithm:**
1.  Give the skateboarder a random shove (draw a random momentum vector).
2.  Simulate them rolling through the bowl using the **Leapfrog Integrator** for a set amount of time.
3.  Because they follow the curves of the math (gravity), they glide along the high-probability ridge instead of stepping off it.
4.  Accept the new location.

*The Problem:* You have to guess how long to let them roll (the `num_steps` parameter). Guess wrong, and they might do a complete U-Turn and roll back to where they started, wasting all the physics calculations.

---

## 2. NUTS (The No-U-Turn Sampler)

NUTS is an algorithm designed to fix the U-Turn problem in HMC, making it a "zero-tuning" sampler.

Instead of guessing a trajectory length, NUTS builds the trajectory dynamically:
1.  It rolls the skateboarder forward and backward in time, doubling the length of the path recursively (1 step, 2 steps, 4 steps, 8 steps...).
2.  At every step, it checks the **Dot Product** between the skateboarder's distance from the start and their current velocity.
3.  The exact moment that dot product becomes negative (meaning the skateboarder is heading back toward where they started), NUTS aborts the simulation.
4.  It picks a random point from the path it just drew.

**Result:** You get perfectly optimized trajectory lengths for every single jump, allowing you to sample incredibly complex distributions with almost 100% acceptance rates and no manual tuning!
