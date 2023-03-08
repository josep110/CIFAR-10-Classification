import torch
import math
import matplotlib.pyplot as plt


N = 2000  # sample count
x = torch.linspace(-math.pi, math.pi, N, dtype=torch.float32) # set range for x
y = torch.sin(x)  # set y axis function in terms of x.
plt.plot(x,y) 
plt.grid()
plt.xlabel('x')          # axis labelling
plt.ylabel('y = f(x)')
plt.show()               # Displays plotted function.
