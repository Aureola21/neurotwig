from neurotwig.core import Value
from neurotwig.visualize import draw_dot


#Manual Backpropogation

#imputs: x1,x2
x1= Value(2.0, label='x1')
x2= Value(0.0, label='x2')

#weights: w1,w2
w1= Value(-3.0, label='w1')
w2= Value(1.0, label='w2')

#bias of the neuron
b=Value(6.881373587019543, label='b')

# neuron computation
x1w1= x1*w1; x1w1.label='x1w1'
x2w2= x2*w2; x2w2.label='x2w2'
x1w1x2w2= x1w1 + x2w2; x1w1x2w2.label='x1w1+x2w2'
n= x1w1x2w2 + b; n.label='n'


# o= n.tanh(); o.label='o'

# tanh activation manually
e_2x = (2 * n).exp()
o = (e_2x - 1) / (e_2x + 1); o.label = 'o'
o.backward_pass()
dot= draw_dot(o)
dot.render('manual_neuron_graph', view=True)
