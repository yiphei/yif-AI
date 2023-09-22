class Value(object):
    def __init__(self, scalar, prevs = None, compute_gradient = None):
        self.scalar = scalar
        self.grad = None
        self.prevs = prevs
        self.compute_gradient = compute_gradient

    def __repr__(self):
        return f"Value({self.scalar})"
    
    def cast_to_value(self, scalar_or_value):
        return scalar_or_value if isinstance(scalar_or_value, Value) else Value(scalar_or_value)

    def __add__(self, other):
        other = self.cast_to_value(other)
        out = Value(self.scalar + other.scalar, prevs=[self, other])
        
        self.compute_gradient = lambda last_grad: last_grad
        other.compute_gradient = lambda last_grad: last_grad
        
        return out
    
    def __relu__(self):
        out  = Value(self.scalar if self.scalar >= 0 else 0, prevs=[self])
        self.compute_gradient = lambda last_grad: last_grad if self.scalar >= 0 else 0

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.scalar ** other, prevs=[self])

        self.compute_gradient = lambda last_grad: last_grad * (other * self.scalar ** (other - 1))
        return out
    
    def __mul__(self,other):
        other = self.cast_to_value(other)
        out = Value(self.scalar * other.scalar, prevs=[self, other])

        self.compute_gradient = lambda last_grad: last_grad * other.scalar
        other.compute_gradient = lambda last_grad: last_grad * self.scalar

        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self,other):
        return other + (-self)
    
    def __truediv__(self,other):
        return self * other ** -1
    
    def __rtruediv__(self,other):
        return other / self
    
    def backward(self, last_grad = None):
        if last_grad is None:
            self.grad = 1
        else:
            self.grad = self.compute_gradient(last_grad)

        if not self.prevs:
            return

        for prev in self.prevs:
            prev.backward(self.grad)
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other