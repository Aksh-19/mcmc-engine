class Node:
    def __init__(self, val, children=()):
        self.val = val                 # The actual number (e.g., 5.0)
        self.grad = 0.0                # The "blame" or derivative (starts at 0)
        self._backward = lambda: None  # The rule for passing blame backward
        self.children = set(children)  # The ingredients that made this node
        
    def __repr__(self):
        return f"Node(val={self.val}, grad={self.grad})"

    # --- ADDITION ---
    def __add__(self, other):
        # If 'other' is just a normal number (like 3), wrap it in a Node
        other = other if isinstance(other, Node) else Node(other)
        
        # Calculate the answer, and record that 'self' and 'other' made it, 
        # parents are indicated as children 
        out = Node(self.val + other.val, (self, other))

        # The rule for passing blame backwards through addition
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # --- MULTIPLICATION ---
    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        
        out = Node(self.val * other.val, (self, other))

        # The rule for passing blame backwards through multiplication
        def _backward():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
        
    # --- NEGATION & SUBTRACTION ---
    def __neg__(self):
        return self * -1.0
        
    def __sub__(self, other):
        return self + (-other)
        
    def __rsub__(self, other):
        return other + (-self)
        
    # --- ADVANCED MATH ---
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Node(self.val ** other, (self,))

        def _backward():
            # derivative of x^n is n * x^(n-1)
            self.grad += (other * (self.val ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        import math
        out = Node(math.exp(self.val), (self,))

        def _backward():
            # derivative of e^x is e^x
            self.grad += out.val * out.grad

        out._backward = _backward
        return out

    def log(self):
        import math
        out = Node(math.log(self.val), (self,))

        def _backward():
            # derivative of ln(x) is 1/x
            self.grad += (1.0 / self.val) * out.grad

        out._backward = _backward
        return out
        
    def __truediv__(self, other):
        # a / b is just a * b^(-1)
        return self * (other ** -1.0)
        
    def __rtruediv__(self, other):
        return other * (self ** -1.0)

    # --- THE BACKWARD SWEEP ---
    def backward(self):
        # 1. Line everyone up from end to beginning (Topological Sort)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2. Start the blame at the very end with 1.0
        self.grad = 1.0
        
        # 3. Tell everyone to pass the blame backwards
        for node in reversed(topo):
            node._backward()
