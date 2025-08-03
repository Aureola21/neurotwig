#Using pytorch 
import torch


x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.881373587019543]).double(); b.requires_grad = True
n= x1*w1 + x2*w2 + b 
o = torch.tanh(n)

print(o.data.item())
o.backward() #backward pass

print()
print(f"x1.grad: {x1.grad.item()}")
print(f"x2.grad: {x2.grad.item()}")
print(f"w1.grad: {w1.grad.item()}")
print(f"w2.grad: {w2.grad.item()}")