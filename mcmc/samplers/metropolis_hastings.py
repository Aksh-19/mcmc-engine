import numpy as np

class MetropolisHastings:
    def __init__(self, target, proposal_scale=1.0, seed=None):
        self.target = target
        self.proposal_scale = proposal_scale
        self.rng = np.random.default_rng(seed)

    def run(self, num_samples, initial_state):
        x = np.array(initial_state, dtype=float)
        if x.ndim > 0: 
            dim = x.shape[0] 
        else: 
            dim = 1
        chain = np.zeros((num_samples, dim))
        chain[0] = x
        accepted = 0
        
        for i in range(1, num_samples):
            noise = self.rng.normal(0, self.proposal_scale, size=dim)
            x_proposed = x + noise

            log_alpha = self.target.log_prob(x_proposed) - self.target.log_prob(x)

            if np.log(self.rng.uniform()) < log_alpha:
                x = x_proposed
                accepted += 1

            chain[i] = x

        rate = accepted / (num_samples-1)
        print(f"Acceptance rate: {rate:.3f}")

        return chain
