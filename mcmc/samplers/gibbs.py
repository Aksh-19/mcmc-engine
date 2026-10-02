import numpy as np


class GibbsSampler:
    def __init__(self, full_conditionals, seed=None):
        self.full_conditionals = full_conditionals
        self.rng = np.random.default_rng(seed)

    def run(self, num_samples, initial_state):
        x = np.array(initial_state, dtype=float).copy()
        dim = len(x)
        chain = np.zeros((num_samples, dim))
        chain[0] = x.copy()

        for i in range(1, num_samples):
            for j in range(dim):
                x[j] = self.full_conditionals[j](x, self.rng)
            chain[i] = x.copy()

        return chain