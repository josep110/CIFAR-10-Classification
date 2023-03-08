import torch
import math
import matplotlib.pyplot as plt

N = 2000 # number of samples
x = torch.linspace(-math.pi, math.pi, N, dtype=torch.float32)
# y = torch.sin(x)
w0 = torch.randn((), dtype=torch.float32)
w1 = torch.randn((), dtype=torch.float32)
w2 = torch.randn((), dtype=torch.float32)
w3 = torch.randn((), dtype=torch.float32)

dat = [w0,w1,w2,w3]

z = torch.tensor(dat)  # creating a tensor from array of random tensors. 
z.requires_grad_(True) # enable gradient calculation.
print(z)

y = torch.dot(z,z)
y.backward()
print(z)
