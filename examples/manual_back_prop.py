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
x1w1= x1*w1; x1w1.label='x1w1'
x2w2= x2*w2; x2w2.label='x2w2'
x1w1x2w2= x1w1 + x2w2; x1w1x2w2.label='x1w1+x2w2'
n= x1w1x2w2 + b; n.label='n'
# o= n.tanh(); o.label='o'
e_2x= (2*n).exp()
o=(e_2x-1)/(e_2x+1); o.label='o'


o.grad=1.0
n.grad= 0.5 #dn/do = 1- tanh^2(n) = 1- o^2
x1w1x2w2.grad=0.5 # dn/d(x1w1+x2w2) = dn/do * dn/d(x1w1+x2w2) = n.grad * 1
b.grad=0.5 # dn/db = dn/do * dn/db = n.grad * 1
x1w1.grad=0.5  # dn/d(x1w1) = dn/do * dn/d(x1w1) = n.grad * 1
x2w2.grad=0.5  # dn/d(x2w2) = dn/do * dn/d(x2w2) = n.grad * 1
x1.grad=x1w1.grad* w1.data # dn/dx1 = dn/d(x1w1) * dw1/dx1 = x1w1.grad * w1.data
x2.grad=x2w2.grad * w2.data # dn/dx2 = dn/d(x2w2) * dw2/dx2 = x2w2.grad * w2.data
w1.grad=x1w1.grad  * x1.data # dn/dw1 = dn/d(x1w1) * dx1w1/dw1 = x1w1.grad * x1.data
w2.grad=x2w2.grad * x2.data # dn/dw2 = dn/d(x2w2) * dx2w2/dw2 = x2w2.grad * x2.data

dot= draw_dot(o)
dot.render('manual_backprop_neuron_graph', view=True)
# o.grad=1.0
# #Performing manual backward pass
# o._backward() #stores do/dn
# n._backward()
# x1w1x2w2._backward()
# b._backward()
# x1w1._backward()
# x2w2._backward ()
# x1._backward()
# x2._backward()
# w1._backward()
# w2._backward()