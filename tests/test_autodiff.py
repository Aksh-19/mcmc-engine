"""Tests for the Reverse-Mode Automatic Differentiation Engine."""

import math
from mcmc.autodiff.reverse import Node

def test_basic_math():
    """Test addition, multiplication, and their derivatives."""
    a = Node(2.0)
    b = Node(3.0)
    
    # f(a,b) = a * b + a
    # df/da = b + 1 = 4.0
    # df/db = a = 2.0
    
    c = a * b + a
    assert c.val == 8.0
    
    c.backward()
    assert a.grad == 4.0
    assert b.grad == 2.0

def test_advanced_math():
    """Test powers, division, and exponents."""
    x = Node(2.0)
    
    # f(x) = (x^3) / 2
    # df/dx = (3/2) * x^2. If x=2, df/dx = 1.5 * 4 = 6.0
    y = (x ** 3) / 2.0
    assert y.val == 4.0
    
    y.backward()
    assert x.grad == 6.0

def test_normal_distribution_gradient():
    """
    Test the gradient of the standard Normal log-probability.
    log_prob(x) = -0.5 * x^2 - log(sqrt(2*pi))
    The mathematical derivative is exactly: -x
    """
    x = Node(3.0)
    
    # Calculate log probability
    constant = math.log(math.sqrt(2 * math.pi))
    log_prob = (-0.5 * (x ** 2)) - constant
    
    log_prob.backward()
    
    # If x = 3.0, the derivative should be -3.0
    assert abs(x.grad - (-3.0)) < 1e-7
    
def test_banana_distribution_gradient():
    """
    Test the gradient of the 2D Banana distribution.
    log_prob = -(1 - x)^2 - 10 * (y - x^2)^2
    """
    x = Node(0.0)
    y = Node(1.0)
    
    # f(x,y) = -(1 - x)^2 - 10 * (y - x^2)^2
    term1 = -((Node(1.0) - x) ** 2)
    term2 = -10.0 * ((y - (x ** 2)) ** 2)
    log_prob = term1 + term2
    
    log_prob.backward()
    
    # Calculus by hand:
    # df/dx = 2(1-x) + 40x(y - x^2)
    # df/dy = -20(y - x^2)
    # At x=0, y=1:
    # df/dx = 2(1) + 0 = 2.0
    # df/dy = -20(1 - 0) = -20.0
    
    assert abs(x.grad - 2.0) < 1e-7
    assert abs(y.grad - (-20.0)) < 1e-7
