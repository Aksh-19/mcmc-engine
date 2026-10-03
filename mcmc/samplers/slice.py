import numpy as np

class SliceSampler:
    def __init__(self, target, w=1.0, seed=None):
        self.target = target
        self.w = w
        self.rng = np.random.default_rng(seed)

    def _slice_sample_1d(self, x, dim_index):
        """
        Performs one univariate slice sampling step for dimension `dim_index`.
        Modifies x[dim_index] in place.
        """
        # 1. Pick slice level (in log-space)
        log_y = self.target.log_prob(x) + np.log(self.rng.uniform())

        # Create a temporary copy of x so we can evaluate log_prob at new points
        x_temp = x.copy()
        current_val = x[dim_index]
        
        # 2. Step out to find [L, R] bounds
        L = current_val - self.w * self.rng.uniform()
        R = L + self.w

        # Step out left
        x_temp[dim_index] = L
        while self.target.log_prob(x_temp) > log_y:
            L -= self.w
            x_temp[dim_index] = L

        # Step out right
        x_temp[dim_index] = R
        while self.target.log_prob(x_temp) > log_y:
            R += self.w
            x_temp[dim_index] = R

        # 3. Shrink and sample
        while True:
            # Pick uniformly from [L, R]
            x_proposed = L + self.rng.uniform() * (R - L)
            x_temp[dim_index] = x_proposed

            # Check if proposed point is inside the slice
            if self.target.log_prob(x_temp) > log_y:
                x[dim_index] = x_proposed  # Accept!
                break
            else:
                # Shrink the window toward the original current_val
                if x_proposed < current_val:
                    L = x_proposed
                else:
                    R = x_proposed

    def run(self, num_samples, initial_state):
        x = np.array(initial_state, dtype=float).copy()
        
        # Handle 1D (scalar) starting points gracefully
        if x.ndim == 0:
            x = np.array([x.item()])
            
        dim = len(x)
        chain = np.zeros((num_samples, dim))
        chain[0] = x.copy()

        for i in range(1, num_samples):
            for j in range(dim):
                self._slice_sample_1d(x, j)
            chain[i] = x.copy()

        return chain