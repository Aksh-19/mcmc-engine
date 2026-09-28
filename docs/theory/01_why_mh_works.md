# Why Metropolis-Hastings Works — Detailed Balance

## The Setup

You have a target distribution π(x) — say, a bell curve. You want to draw
samples from it. But you can't sample directly (maybe it's a weird shape,
or high-dimensional, or you only know it up to a constant).

MH gives you a way: build a random walk that, after enough steps, visits
each point x proportionally to π(x).

## The Algorithm (Recap)

1. Start at some point x
2. Propose x' = x + noise
3. Compute α = π(x') / π(x)
4. If α ≥ 1: accept (move to x')
   If α < 1: accept with probability α, else stay at x
5. Repeat

## Why This Works: Detailed Balance

Imagine the chain has been running forever. It visits point x with frequency
f(x). We want f(x) = π(x).

**Detailed balance** says: for any two points x and x', the "flow" from
x → x' equals the flow from x' → x.

    π(x) · T(x → x') = π(x') · T(x' → x)

Where T(x → x') is the transition probability (probability of moving from
x to x' in one step).

### Proof sketch for symmetric proposals

For a symmetric random walk (where proposing x' from x is equally likely as
proposing x from x'), the transition probability is:

    T(x → x') = q(x'|x) · min(1, π(x')/π(x))

Since q is symmetric: q(x'|x) = q(x|x'). Call this q.

Case 1: π(x') ≥ π(x)
    Left side:  π(x) · q · 1 = q · π(x)
    Right side: π(x') · q · (π(x)/π(x')) = q · π(x)    ✓ equal

Case 2: π(x') < π(x)
    Left side:  π(x) · q · (π(x')/π(x)) = q · π(x')
    Right side: π(x') · q · 1 = q · π(x')               ✓ equal

Both cases balance. Therefore π is the stationary distribution of the chain.

## What This Means Intuitively

The accept/reject rule is carefully designed so that:
- If π(x) is twice as large as π(x'), the chain spends twice as much time at x
- This holds for every pair of points simultaneously
- After enough steps, the histogram of visited points matches π exactly

## What I Verified With My Code

- Sampled 50,000 points from Normal(0,1) — histogram matches the bell curve
- Sampled from Beta(2,5) — histogram matches the skewed shape
- Sampled from the 2D Banana — scatter plot follows the curved valley
- All tests pass: mean and standard deviation match theoretical values

The math works. The code works. Detailed balance is real.
