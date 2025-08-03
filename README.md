# 🌿 neurotwig

A minimal neural network engine built from scratch using pure Python — with automatic differentiation, graph visualization, and a micro MLP.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), `neurotwig` is educational, lightweight, and completely hackable.

---

## 🧠 Features

- `Value` class with backward propagation (autograd)
- Support for basic operations: `+`, `-`, `*`, `/`, `**`, `tanh`, `exp`
- Visualize computation graph using [Graphviz](https://graphviz.org/)
- Manual single neuron and multi-layer perceptron (MLP)
- Training loop with gradient descent
- No external ML libraries

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/your-username/neurotwig.git
cd neurotwig
pip install -e .
```

> This will install `neurotwig` as an editable package for local development.

---

## 🗂 Project Structure

```
neurotwig/
├── neurotwig/              # Core library code
│   ├── __init__.py
│   ├── core.py             # Value class + autograd
│   ├── net_struc.py        # Neuron, Layer, MLP
│   └── visualize.py        # Computation graph rendering
│
├── examples/               # Example scripts
│   ├── basic_ops.py        # Basic math and graph visualization
│   ├── manual_neuron.py    # Manual neuron with tanh
│   └── manual_back_prop.py # Manual back propogation
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Examples

### Visualizing Basic Computation Graph

```python
from neurotwig.core import Value
from neurotwig.visualize import draw_dot

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = a * b
c.label = 'c'

draw_dot(c).render('graph', format='png', cleanup=False)
```

### Training a Tiny Neural Network

```python
from neurotwig.model import MLP

mlp = MLP(3, [4, 4, 1])
x = [2.0, 3.0, -1.0]
out = mlp(x)
print(out)
```

---

## 📚 Learn More

This project is perfect for:

- Understanding how backpropagation works
- Building your own neural networks
- Learning Python classes and autograd logic
- Visualizing how gradients flow

---

## 🔖 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Credits

Inspired by [micrograd](https://github.com/karpathy/micrograd) by [Andrej Karpathy](https://github.com/karpathy).
