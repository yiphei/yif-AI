class Value(object):
    def __init__(self, scalar, prevs = None, compute_prev_gradients = None):
        self.scalar = scalar
        self.grad = 0
        self.prevs = prevs
        self.compute_prev_gradients = compute_prev_gradients

    def __repr__(self):
        return f"Value({self.scalar})"
    
    def cast_to_value(self, scalar_or_value):
        return scalar_or_value if isinstance(scalar_or_value, Value) else Value(scalar_or_value)

    def __add__(self, other):
        other = self.cast_to_value(other)
        out = Value(self.scalar + other.scalar, prevs=[self, other])

        def compute_prev_gradients(last_grad):
            self.grad += last_grad
            other.grad += last_grad
            self.backward(is_first = False)
            other.backward(is_first=False)
        
        out.compute_prev_gradients = compute_prev_gradients
        return out
    
    def relu(self):
        out  = Value(self.scalar if self.scalar >= 0 else 0, prevs=[self])

        def compute_prev_gradients(last_grad):
            self.grad += last_grad if self.scalar >= 0 else 0
            self.backward(is_first = False)

        out.compute_prev_gradients = compute_prev_gradients
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.scalar ** other, prevs=[self])
        self.compute_prev_gradients = lambda last_grad: last_grad * (other * self.scalar ** (other - 1))
        return out
    
    def __mul__(self,other):
        if other == self:
            return self ** 2
        other = self.cast_to_value(other)
        out = Value(self.scalar * other.scalar, prevs=[self, other])

        def compute_prev_gradients(last_grad):
            self.grad += last_grad * other.scalar
            other.grad += last_grad * self.scalar
            self.backward(is_first = False)
            other.backward(is_first=False)                
        
        out.compute_prev_gradients = compute_prev_gradients

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
    
    def backward(self, is_first = True):
        if is_first:
            self.grad = 1

        if self.compute_prev_gradients:
            self.compute_prev_gradients(self.grad)
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other