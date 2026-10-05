# 🧠 mcmc-engine

A fully functional MCMC sampling library built from scratch in Python — **numpy and matplotlib only**.

No PyMC. No Stan. No JAX. No PyTorch. **You are the inference engine.**

## What is this?

A mini-Stan: give it any probabilistic model, and it finds the posterior using physics-inspired sampling. Built for learning, built for understanding, built to make interviewers say *"wait, you built this from scratch?"*

## Samplers

| Sampler | Status | Description |
|---------|--------|-------------|
| Metropolis-Hastings | ✅ | The foundational random-walk sampler |
| Gibbs | ✅ | Coordinate-wise sampling with exact full conditionals |
| Slice | ✅ | Adaptive, tuning-free univariate/multivariate sampling |
| HMC | ✅ | Hamiltonian Monte Carlo — physics meets statistics |
| NUTS | 🔲 | No-U-Turn Sampler — the algorithm inside Stan |

## Architecture

```
mcmc-engine/
├── mcmc/
│   ├── distributions/    # Target distributions (priors, likelihoods)
│   ├── samplers/          # MH, Gibbs, Slice, HMC, NUTS
│   ├── diagnostics/       # R-hat, ESS, trace plots, autocorrelation
│   └── autodiff/          # Forward-mode (dual numbers) + reverse-mode (tape)
├── tests/                 # pytest test suite
├── examples/              # Jupyter notebooks demonstrating usage
└── docs/theory/           # Derivations, intuition, "aha moments"
```

## Quick Start

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

```python
from mcmc.samplers import MetropolisHastings
from mcmc.distributions import Normal
from mcmc import diagnostics

# Define your target distribution
target = Normal(mu=0, sigma=1)

# Sample from it
sampler = MetropolisHastings(target, proposal_scale=0.5)
chain = sampler.run(num_samples=10_000, initial_state=0.0)

# Check convergence
diagnostics.trace_plot(chain)
diagnostics.effective_sample_size(chain)
```

## Roadmap

| Phase | Topic |
|-------|-------|
| **0** | Repo setup & project skeleton ✅ |
| **1** | Probability foundations + Metropolis-Hastings ✅ |
| **2** | Gibbs sampling + Slice sampling ✅ |
| **3** | Diagnostics suite (R-hat, ESS, trace plots) ✅ |
| **4** | Automatic differentiation engine (forward + reverse) ✅ |
| **5** | Hamiltonian Monte Carlo ✅ |
| **6** | NUTS (No-U-Turn Sampler) |
| **7** | Capstone apps (change-point detection, hierarchical A/B testing, GP regression) |
| **8** | Polish, docs & showcase |

## Dependencies

- `numpy` — numerical computation
- `matplotlib` — visualization only
- `pytest` — testing

## License

MIT
