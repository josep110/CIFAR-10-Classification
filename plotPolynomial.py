import torch
import math
import matplotlib.pyplot as plt

N = 2000  # sample count
x = torch.linspace(-math.pi, math.pi, N, dtype=torch.float32) # set range for x

w0 = torch.randn((), dtype=torch.float32)
w1 = torch.randn((), dtype=torch.float32)  # random weights
w2 = torch.randn((), dtype=torch.float32)
w3 = torch.randn((), dtype=torch.float32)

y_hat = w0 + w1*x + w2*x**2 + w3*x**3      # Polynomial of degree 3.

plt.plot(x,y_hat) 
plt.grid()
plt.xlabel('x')          # axis labelling
plt.ylabel('y = f(x)')
plt.show()               # Displays plotted function.