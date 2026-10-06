# MCMC Theory & Architecture

This directory contains the theoretical background, mathematical explanations, and engineering design decisions behind the MCMC engine. These documents are designed to serve as an interactive textbook explaining *how* and *why* these complex probabilistic algorithms work.

## Core Concepts & Diagnostics
* `01_why_mh_works.md`: Explains the concept of Detailed Balance and the mathematical proof behind the Metropolis-Hastings acceptance criteria.
* `02_which_sampler.md`: A comprehensive guide comparing Metropolis-Hastings, Gibbs, and Slice sampling, explaining when to use each based on the geometry of the target distribution.
* `03_diagnostics.md`: The math behind chain convergence, detailing Gelman-Rubin ($\hat{R}$), Effective Sample Size (ESS), and Autocovariance.

## The Physics Engine & Automatic Differentiation
* `04_autodiff.md`: Explains the architecture of our custom Reverse-Mode Automatic Differentiation framework, highlighting how the `Node` computation graph backpropagates gradients.
* `05_physics_samplers.md`: Details how Hamiltonian Monte Carlo (HMC) and the No-U-Turn Sampler (NUTS) utilize simulated physical momentum, leapfrog integration, and recursive tree-building to efficiently explore high-dimensional probability spaces.

## Real-World Applications (Capstones)
* `06_capstone_geiger.md`: Breaks down the Inverse-Square Law, Poisson statistics, and the critical MCMC "Log-Parameterization" trick used to map radiation sources in 3D without crashing the physics engine.
* `07_capstone_cyberattack.md`: Explains Bayesian Change-Point Detection, the "Vanishing Gradient" problem, and the elegant "Continuous Sigmoid Relaxation" trick used to make discrete calendar days differentiable for NUTS.
