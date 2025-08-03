from core import Value
import random

class Neuron:
    def __init__(self, n_input):
        self.w=[Value(random.uniform(-1,1)) for _ in range(n_input)]
        self.b= Value(random.uniform(-1,1))

    def __call__(self,x):
        #w*(x + b)
        #list(zip(self.w,x)) creates a list of tuples of w and x as [(w1,x1),(w2,x2),...]
        y = sum ((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out = y.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]  #returns a list of all the parameters of the neuron (weights and bias)

class Layer:

    def __init__(self,n_input,n_output):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self,x):
        #x is a list of inputs
        #we will call each neuron in the layer with the same input x
        #and return a list of outputs from each neuron
        output = [n(x) for n in self.neurons]
        return output[0] if len(output) == 1 else output
    def parameters(self):
        params=[]
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params  #returns a list of all the parameters of the layer (weights and bias

class MLP:
    def __init__(self,n_int,n_outs_list): #n_outs_list is a list of number of outputs for each layer
        #n_int is the number of inputs to the first layer
        sz = [n_int]+n_outs_list
        #sz is a list of sizes of each layer
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_outs_list))]
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)    
        return x
    
    def parameters(self):
        params=[]
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params  #returns a list of all the parameters of the MLP (weights and bias)