from .base import Distribution
import numpy as np

class Normal(Distribution):        
    def __init__(self, mu, sigma):  
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x):         
        return (-0.5 * ((x - self.mu) / self.sigma)**2 - np.log(self.sigma) - 0.5*np.log(2*np.pi))

class Beta(Distribution):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def log_prob(self, x):
        if x <= 0 or x >= 1:
            return -np.inf              # impossible region
        return (self.alpha-1)*np.log(x) + (self.beta-1)*np.log(1-x)

class Banana(Distribution):
    def __init__(self, a=1.0, b=10.0):
        self.a = a
        self.b = b

    def log_prob(self, x):
        return (-(self.a - x[0])**2 - self.b*(x[1] - x[0]**2)**2)