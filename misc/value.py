class Value(object):
    def __init__(self, scalar, prevs = None, compute_prev_gradients = None):
        self.scalar = scalar
        self.grad = 0
        self.prevs = prevs
        self.compute_prev_gradients = compute_prev_gradients
        self.count = 0

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
        
        self.count += 1
        other.count += 1
        out.compute_prev_gradients = compute_prev_gradients
        return out
    
    def tanh(self):
        import math

        x = self.scalar
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, prevs= [self])
        
        def compute_prev_gradients(last_grad):
            self.grad += (1 - t**2) * last_grad
        
        self.count += 1
        out.compute_prev_gradients = compute_prev_gradients
        return out
    
    def relu(self):
        out  = Value(self.scalar if self.scalar >= 0 else 0, prevs=[self])

        def compute_prev_gradients(last_grad):
            self.grad += last_grad if self.scalar >= 0 else 0

        self.count += 1
        out.compute_prev_gradients = compute_prev_gradients
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.scalar ** other, prevs=[self])

        def compute_prev_gradients(last_grad):
            self.grad += last_grad * (other * self.scalar ** (other - 1))

        self.count += 1
        out.compute_prev_gradients = compute_prev_gradients
        return out
    
    def __mul__(self,other):
        other = self.cast_to_value(other)
        out = Value(self.scalar * other.scalar, prevs=[self, other])

        def compute_prev_gradients(last_grad):
            self.grad += last_grad * other.scalar
            other.grad += last_grad * self.scalar
        
        self.count += 1
        other.count += 1

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
        return (self  ** -1) * other
    
    def backward(self):
        from queue import Queue

        self.grad = 1

        q = Queue()
        q.put(self)
        while not q.empty():
            value = q.get()
            if value.compute_prev_gradients:
                value.compute_prev_gradients(value.grad)
            
                for prev in value.prevs:
                    prev.count -= 1
                    if prev.count <= 0:
                        q.put(prev)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other