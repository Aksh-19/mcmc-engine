"""Automatic differentiation engine.

This module provides exact, automatic computation of gradients — the same
idea behind PyTorch and JAX, but built from scratch in ~200 lines.

Planned components:
    - Dual — dual number class for forward-mode autodiff
    - Tape — computation graph for reverse-mode autodiff (backpropagation)
    - grad() — user-facing API to compute gradients of arbitrary functions

Forward-mode: one derivative per forward pass (good for few parameters).
Reverse-mode: ALL derivatives in one backward pass (essential for HMC/NUTS).
"""
