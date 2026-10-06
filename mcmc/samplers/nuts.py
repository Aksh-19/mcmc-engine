import numpy as np
from mcmc.samplers.hmc import get_gradient_and_energy

def leapfrog(q, p, grad_U, step_size, target):
    """Takes a single Leapfrog step in time."""
    p_new = p - (step_size / 2.0) * grad_U
    q_new = q + step_size * p_new
    U_new, grad_U_new = get_gradient_and_energy(target, q_new)
    p_new = p_new - (step_size / 2.0) * grad_U_new
    return q_new, p_new, grad_U_new, U_new

def is_u_turn(q_left, q_right, p_left, p_right):
    """Detects if the trajectory is curving back on itself."""
    diff = q_right - q_left
    # If dot product is negative, momentum is pointing in opposite direction of the distance
    return (np.dot(diff, p_left) < 0) or (np.dot(diff, p_right) < 0)

def build_tree(q, p, grad_U, log_u, v, j, step_size, target, H0, rng):
    """
    Recursively builds a binary tree of leapfrog trajectories.
    v is direction (-1 for backward in time, 1 for forward).
    j is the depth of the tree (2^j steps).
    """
    # Base case: Take exactly 1 leapfrog step
    if j == 0:
        q_prime, p_prime, grad_U_prime, U_prime = leapfrog(q, p, grad_U, v * step_size, target)
        H_prime = U_prime + 0.5 * np.sum(p_prime**2)
        
        # Is this new point valid? (Is its energy acceptable?)
        n_prime = 1 if log_u <= -H_prime else 0
        
        # Did the physics explode? (Divergence check)
        s_prime = 1 if log_u < 1000.0 - H_prime else 0 
        
        return q_prime, p_prime, grad_U_prime, q_prime, p_prime, grad_U_prime, q_prime, n_prime, s_prime

    # Recursive case: Build a tree of depth j-1, then build another one attached to it
    q_m, p_m, grad_m, q_p, p_p, grad_p, q_prime, n_prime, s_prime = build_tree(
        q, p, grad_U, log_u, v, j - 1, step_size, target, H0, rng)
    
    if s_prime == 1:
        if v == -1: # Build backwards
            q_m, p_m, grad_m, _, _, _, q_prime2, n_prime2, s_prime2 = build_tree(
                q_m, p_m, grad_m, log_u, v, j - 1, step_size, target, H0, rng)
        else:       # Build forwards
            _, _, _, q_p, p_p, grad_p, q_prime2, n_prime2, s_prime2 = build_tree(
                q_p, p_p, grad_p, log_u, v, j - 1, step_size, target, H0, rng)
        
        # Metropolis accept/reject to pick a point from the newly built half of the tree
        if n_prime2 > 0:
            if rng.uniform() < (n_prime2 / (n_prime + n_prime2)):
                q_prime = q_prime2
        
        n_prime += n_prime2
        
        # Check if the FULL joined tree makes a U-Turn
        s_prime = s_prime2 and int(not is_u_turn(q_m, q_p, p_m, p_p))
        
    return q_m, p_m, grad_m, q_p, p_p, grad_p, q_prime, n_prime, s_prime


class NUTSSampler:
    def __init__(self, target, step_size=0.1, max_depth=10, seed=None):
        self.target = target
        self.step_size = step_size
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

    def run(self, num_samples, initial_state):
        q = np.array(initial_state, dtype=float).copy()
        if q.ndim == 0:
            q = np.array([q.item()])
            
        dim = len(q)
        chain = np.zeros((num_samples, dim))
        chain[0] = q.copy()
        
        for i in range(1, num_samples):
            # 1. Random Momentum
            p = self.rng.normal(0, 1, size=dim)
            U_current, grad_U = get_gradient_and_energy(self.target, q)
            H_current = U_current + 0.5 * np.sum(p**2)
            
            # 2. Draw slice variable to define "acceptable" energy levels
            log_u = np.log(self.rng.uniform()) - H_current
            
            # Initialize tree ends
            q_m, q_p = q.copy(), q.copy()
            p_m, p_p = p.copy(), p.copy()
            grad_m, grad_p = grad_U.copy(), grad_U.copy()
            
            j = 0
            n = 1
            s = 1
            
            # 3. Double the tree size until a U-turn happens (s == 0)
            while s == 1 and j < self.max_depth:
                # Randomly choose to build forward (+1) or backward (-1) in time
                v = self.rng.choice([-1, 1])
                
                if v == -1:
                    q_m, p_m, grad_m, _, _, _, q_prime, n_prime, s_prime = build_tree(
                        q_m, p_m, grad_m, log_u, v, j, self.step_size, self.target, H_current, self.rng)
                else:
                    _, _, _, q_p, p_p, grad_p, q_prime, n_prime, s_prime = build_tree(
                        q_p, p_p, grad_p, log_u, v, j, self.step_size, self.target, H_current, self.rng)
                
                # If the new tree is valid, maybe accept the proposed point
                if s_prime == 1:
                    if self.rng.uniform() < min(1, n_prime / n):
                        q = q_prime
                
                n += n_prime
                # Detect U-turn across the entire span of the new tree
                s = s_prime and int(not is_u_turn(q_m, q_p, p_m, p_p))
                j += 1
                
            chain[i] = q.copy()
            
        return chain
