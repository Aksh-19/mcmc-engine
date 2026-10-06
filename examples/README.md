# MCMC Engine Examples

This directory contains all the executable demo scripts built during the development of the MCMC engine. Each script demonstrates a specific feature, sampler, or real-world application. All generated visual outputs are saved to the `plots/` directory.

## Basic Samplers & Mechanics
* `01_sampling_distributions.py`: Demonstrates the basic Metropolis-Hastings sampler on standard target distributions.
* `02_proposal_tuning.py`: Visualizes how tuning the proposal variance affects acceptance rates and exploration.
* `03_sampler_comparison.py`: Compares Metropolis-Hastings, Gibbs, and Slice sampling on a 2D Gaussian target.
* `04_diagnostics_demo.py`: Demonstrates MCMC health metrics including trace plots, Autocorrelation, Gelman-Rubin (R-hat), and Effective Sample Size (ESS).

## Physics-Based Samplers (Using Autodiff)
* `05_hmc_demo.py`: Runs Hamiltonian Monte Carlo (HMC) using leapfrog integration and our custom Reverse-Mode Autodiff engine to efficiently explore the highly curved Banana distribution.
* `06_nuts_demo.py`: Demonstrates the No-U-Turn Sampler (NUTS) dynamically auto-tuning its trajectory lengths via recursive tree-building to prevent U-turns.

## Capstone Real-World Applications
* `07_geiger_counter.py`: **Capstone 1** — Finds a hidden radioactive source in 3D space using noisy Poisson counts and the Inverse-Square Law.
* `07_geiger_multiple_sources.py`: **Capstone 1 (Advanced)** — Maps two radioactive sources simultaneously by exploring an 8-dimensional posterior space.
* `07_geiger_heatmap.py`: Generates a continuous, glowing 2D Seaborn KDE heatmap of the Geiger counter posterior density over a physical sensor grid.
* `07_change_point.py`: **Capstone 2** — Detects a stealth cyber-attack hidden in massive daily server noise using Bayesian Change-Point Detection and continuous sigmoid relaxation.

## Bonus Visualizations
* `08_sampler_comparison_jointplot.py`: Generates a paper-quality Seaborn JointGrid comparing the KDE contours and marginals of MH, HMC, and NUTS.

---

### Running the Examples
All scripts should be executed from the root directory of the project to ensure imports work correctly:
```bash
python examples/07_geiger_heatmap.py
```
