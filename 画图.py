import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
y = torch.asin(x) + 0.2*torch.rand(x.size())

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
