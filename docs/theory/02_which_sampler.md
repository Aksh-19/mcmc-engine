# Which Sampler Should I Use? (MH vs Gibbs vs Slice)

When writing an MCMC engine, you have multiple samplers at your disposal. They all guarantee detailed balance (meaning they all eventually sample the true distribution), but they get there in very different ways.

Here is the cheat sheet on when to use which.

---

## 1. Metropolis-Hastings (MH)

**How it works:** Proposes a random jump in *all dimensions at once*. If it's better, accept. If it's worse, maybe accept.
**Pros:** 
- Incredibly simple to code.
- Only requires a `log_prob` function.
**Cons:**
- **The Curse of Dimensionality:** In 100 dimensions, a random jump is almost guaranteed to be terrible. Acceptance rates plummet.
- **Tuning:** You *must* tune the proposal scale (the step size). If it's too small, it takes forever. If it's too big, you reject everything.

**Verdict:** Use for 1D or 2D toy problems, or as a building block for other algorithms. Rarely used vanilla in production for high dimensions.

---

## 2. Gibbs Sampling

**How it works:** Updates *one dimension at a time*, freezing all others. It draws an exact sample from the "full conditional" distribution.
**Pros:**
- **No tuning!** There is no step size to guess.
- **100% Acceptance:** Because you draw exactly from the mathematical conditional, you never reject a sample.
- Scales beautifully to high dimensions (if variables aren't too highly correlated).
**Cons:**
- **Math heavy:** You, the human, must do calculus/algebra to figure out the full conditional equations beforehand. 
- If you have a weird, non-standard distribution, the full conditionals might be impossible to write down.

**Verdict:** Use when your model relies on standard distributions (like Gaussians, Betas, etc.) where "conjugate priors" let you easily calculate the full conditional formulas.

---

## 3. Slice Sampling

**How it works:** Picks a random "slice" horizontally across the probability hill, then picks a point uniformly from that slice. Updates one coordinate at a time like Gibbs.
**Pros:**
- **No tuning!** The window width `w` is adaptive; if you guess wrong, the algorithm shrinks the window automatically.
- **No math required:** Unlike Gibbs, you don't need to derive full conditionals. You just need a `log_prob` function.
- Rarely gets stuck, even in weirdly shaped distributions.
**Cons:**
- Each step requires evaluating the `log_prob` multiple times (while stepping out and shrinking), making it computationally slower per-iteration than MH or Gibbs.

**Verdict:** The ultimate robust fallback. If you don't want to tune MH step sizes, and you can't do the math for Gibbs, use Slice. 

---

### Summary Table

| Feature | Metropolis-Hastings | Gibbs | Slice |
|---------|---------------------|-------|-------|
| **Requires Tuning?** | Yes (step size) | No | No (adaptive) |
| **Requires Math Derivations?** | No | Yes (conditionals) | No |
| **Acceptance Rate** | Varies (aim for 23%) | 100% | 100% |
| **High Dimension Perf** | Poor | Excellent | Good |
