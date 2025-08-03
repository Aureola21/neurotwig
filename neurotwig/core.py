import math

class Value:

    def __init__(self, data, _children=(),_op='',label=''):
        self.data=data
        self.grad=0
        self._backward= lambda: None #Empty function for backward pass
        self._prev = set(_children)
        self._op=_op
        self.label=label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        sum= Value(self.data+other.data, (self,other),"+")
        def _backward():
            self.grad += 1.0 * sum.grad
            other.grad += 1.0 * sum.grad
        sum._backward=_backward
        return sum
    def __radd__(self,other): #other+self
        return self + other  #This is called when python cannot do a.__add__(b
    
    def __mul__(self,other):
        #a*b ----> a.__mul__(b)
        other = other if isinstance(other, Value) else Value(other)
        pdt = Value(self.data*other.data, (self, other),"*")

        def _backward():
            self.grad += other.data*pdt.grad
            other.grad += self.data*pdt.grad
        pdt._backward = _backward
        return pdt
    
    def __rmul__(self,other): #other*self
        return self * other 
    # for a*b, rmul is called when python cannot do a.__mul__(b), hence it calls b.__rmul__(a)

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"power must be integer or float"
        pwr=self.data**other
        out=Value(pwr,(self,),f"**{other}")
        def _backward():
            self.grad += other*(self.data**(other - 1)) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self,other):
        return self*other**-1

    def __neg__(self):
        return self * -1
    def __sub__(self,other): #self-other
        return self + (-other)
    
    def exp(self):
        x=self.data
        t= math.exp(x)
        out=Value(t,(self,),"exp")

        def _backward():
            self.grad+= t * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        x= self.data
        t =(math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out= Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward_pass(self):
            
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        #Now we have a topological order of nodes in the graph
        self.grad=1.0
        #We will now traverse the graph in reverse order
        #and call the backward function for each node
        for node in reversed(topo):
            node._backward()

