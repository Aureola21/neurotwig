from neurotwig.core import Value
from neurotwig.visualize import draw_dot


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

L.backward_pass()
dot = draw_dot(L)
dot.render('basic_ops_graph', view=True)
