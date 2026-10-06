# How Do We Know When MCMC is Done? (Diagnostics)

MCMC is a stochastic (random) process. If you run a sampler, it will *always* give you numbers. The dangerous part is knowing whether you can trust those numbers. 

Here are the three tools we use to verify convergence.

---

## 1. Trace Plots (The "Eye Test")
A trace plot is a simple line graph of the parameter values over time.
*   **Good:** It looks like a "fuzzy caterpillar" bouncing rapidly around a stable mean.
*   **Bad:** It slowly wanders up or down, or gets stuck in one place for long periods. 

## 2. Effective Sample Size (ESS)
Because MCMC steps are generated from the *previous* step, the samples are highly correlated. 
If you draw 10,000 samples, you don't actually have 10,000 independent pieces of information. 

**ESS** calculates how many *truly independent* samples your chain is worth. 
It does this by measuring **Autocorrelation**—how similar step $t$ is to step $t+1$, $t+2$, etc. It heavily penalizes slow-moving chains. 
*   *Metropolis-Hastings* might run 10,000 steps but only give an ESS of 200.
*   *NUTS* might run 1,000 steps and give an ESS of 1,000!

## 3. Gelman-Rubin Statistic ($\hat{R}$ / R-hat)
This is the single most important diagnostic. Sometimes, a trace plot looks perfectly fine, but the sampler is actually trapped on one peak of a multi-peak distribution.

**How it works:**
1. You run multiple chains (e.g., 4) from completely different starting locations.
2. You measure the **Within-chain variance (W)**: How wide is the "caterpillar" of a single chain?
3. You measure the **Between-chain variance (B)**: How far apart are the centers of the 4 caterpillars?
4. If they have all found the true distribution, their centers will overlap, making $B \approx 0$. 

$\hat{R} = \sqrt{\frac{\text{Var}(\theta)}{W}}$

**The Rule of Thumb:**
*   $\hat{R} \approx 1.0$: Excellent. All chains found the exact same shape.
*   $\hat{R} > 1.05$: **Danger!** The chains disagree. Do not trust your results; run the sampler longer or use a better sampler (like NUTS).
