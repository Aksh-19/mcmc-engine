import numpy as np
from mcmc.autodiff.reverse import Node

def get_gradient_and_energy(target, q_array):
    """Uses our Autodiff engine to calculate Potential Energy and its slope."""
    # 1. Convert standard numpy array into our Autodiff Nodes
    q_nodes = [Node(val) for val in q_array]
    
    # 2. Run the forward pass to get the log probability
    if len(q_nodes) == 1:
        log_p_node = target.log_prob(q_nodes[0])
    else:
        log_p_node = target.log_prob(q_nodes)
    
    # 3. Run the backward pass to calculate the gradients
    log_p_node.backward()
    
    # 4. Extract the gradients back into a standard numpy array
    grads = np.array([node.grad for node in q_nodes])
    
    # Physics definitions: 
    # Potential Energy U = -log(probability)
    # Force = -gradient of log(probability)
    U = -log_p_node.val
    grad_U = -grads
    
    return U, grad_U

class HMCSampler:
    def __init__(self, target, step_size=0.1, num_steps=10, seed=None):
        self.target = target
        self.step_size = step_size
        self.num_steps = num_steps
        self.rng = np.random.default_rng(seed)

    def run(self, num_samples, initial_state):
        q = np.array(initial_state, dtype=float).copy()
        if q.ndim == 0:
            q = np.array([q.item()])
            
        dim = len(q)
        chain = np.zeros((num_samples, dim))
        chain[0] = q.copy()
        accepted = 0

        for i in range(1, num_samples):
            # 1. Give the system a random kick (initial momentum)
            p = self.rng.normal(0, 1, size=dim)
            
            # 2. Measure starting Energy
            U_current, grad_U = get_gradient_and_energy(self.target, q)
            K_current = 0.5 * np.sum(p**2)
            H_current = U_current + K_current
            
            # --- START PHYSICS SIMULATION (Leapfrog Integrator) ---
            q_new = q.copy()
            p_new = p.copy()
            
            # Take an initial HALF step of momentum based on the current slope
            p_new = p_new - (self.step_size / 2.0) * grad_U
            
            for step in range(self.num_steps):
                # Take a FULL step of position using our current momentum
                q_new = q_new + self.step_size * p_new
                
                # Check the new slope at our new position
                U_new, grad_U_new = get_gradient_and_energy(self.target, q_new)
                
                # If we aren't at the very end, take a FULL step of momentum
                if step != self.num_steps - 1:
                    p_new = p_new - self.step_size * grad_U_new
            
            # Take a final HALF step of momentum to finish the simulation
            p_new = p_new - (self.step_size / 2.0) * grad_U_new
            # --- END PHYSICS SIMULATION ---
            
            # Reverse momentum to ensure the math remains perfectly symmetric
            p_new = -p_new
            
            # 3. Measure final Energy
            K_new = 0.5 * np.sum(p_new**2)
            H_new = U_new + K_new
            
            # 4. Metropolis Accept/Reject Step
            # If simulation was perfect, H_new == H_current, and alpha == 1 (100% accept)
            alpha = np.exp(H_current - H_new)
            
            if self.rng.uniform() < alpha:
                q = q_new
                accepted += 1
                
            chain[i] = q.copy()

        rate = accepted / (num_samples - 1)
        print(f"HMC Acceptance rate: {rate:.3f}")
        return chain