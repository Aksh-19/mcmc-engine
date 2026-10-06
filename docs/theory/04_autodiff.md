# Automatic Differentiation: The Engine Under the Hood

To use advanced samplers like HMC or NUTS, we need to know the *slope* (the gradient) of the probability distribution at any given point. 

## Why not use standard calculus?
1.  **By Hand:** You can derive the derivatives on a whiteboard. But if you change your statistical model, you have to do the math all over again.
2.  **Finite Differences (Guessing):** You can calculate $\frac{f(x + 0.001) - f(x)}{0.001}$. But if you have 1,000 parameters, you have to run your model 1,000 times just to find the slope once. It is painfully slow and numerically unstable.

## The Solution: Reverse-Mode Autodiff
This is the exact same technology that powers PyTorch, TensorFlow, and large language models. It applies the **Chain Rule of Calculus** automatically across a computation graph.

### 1. The Forward Pass (Building the Receipt)
When you wrap a number in our `Node` class and do math with it (e.g., `c = a + b`), the computer doesn't just calculate the answer. It creates a "receipt". 
The new node `c` remembers:
*   Its value.
*   Who created it (`a` and `b`).
*   The exact mathematical rule to pass derivatives backward through addition.

### 2. The Backward Pass (Passing the Blame)
When you finally calculate your log-probability, you have a massive tree of operations. You call `.backward()` on the final answer.

The algorithm works backward from the answer to the inputs:
*   It looks at the final node and asks, "How did you get here?"
*   It traces the exact operations backward, multiplying the local derivatives (chain rule).
*   By the time it reaches the original inputs (your parameters), it has deposited the exact mathematical gradient into them.

**The Magic:** It can calculate the exact derivative for 1,000 parameters in a *single backward sweep*. This makes high-dimensional physics-based sampling possible!
