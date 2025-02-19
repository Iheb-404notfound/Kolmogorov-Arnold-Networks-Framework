import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from src.kan import KAN_Layer

class KanNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, knots, generator_function):
        super(KanNetwork, self).__init__()
        self.kan1 = KAN_Layer(input_size, hidden_size, knots, generator_function)
        self.kan2 = KAN_Layer(hidden_size, output_size, knots, generator_function)
    
    def forward(self, x):
        x = self.kan1(x)
        x = self.kan2(x)
        return x


input_size = 2
hidden_size = 3
output_size = 1
knots = 3
generator_function = lambda x, mu = 0, s = 1: torch.exp(-(x-mu)**2/(2*s**2))/(s*torch.math.sqrt(2*torch.pi))

features = torch.stack([torch.linspace(-2, 2, 100), torch.linspace(-1, 3, 100)], dim=-1)
x = features[:, 0]
y = features[:, 1]
labels = torch.sin(-x).data + torch.cos(y).data
labels = labels.unsqueeze(-1)




# Model, Optimizer, and Loss Function
model = KanNetwork(input_size, hidden_size, output_size, knots, generator_function)
optimizer = optim.Adam(model.parameters(), lr=0.001)
history = []


# Training Loop
print("Training Loop")
for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = model(features)
    #print(y_pred.shape, labels.shape)
    loss = nn.MSELoss()(y_pred, labels)
    history.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}", end='\r')
    loss.backward()
    optimizer.step()
print(f"Epoch {epoch}, Loss: {loss.item()}")



# Plotting the results
plt.rcParams.update({'font.size': 16})
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(8, 5))
plt.grid(linestyle='dotted')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.plot(x.squeeze(-1).detach().numpy(), labels.squeeze(-1).detach().numpy(), 
         label=r'$\sin(-x) + \cos(y)$',
         linestyle="--",
         linewidth=1.8,
         color='red'
        )
plt.plot(x.squeeze(-1).detach().numpy(), y_pred.squeeze(-1).detach().numpy(), 
         label=r'$f(x)$',
         linewidth=1.5,
         color='blue'
        )
plt.legend(ncol=2, loc=2)
plt.tight_layout()
plt.savefig("sin-x.svg", dpi=300)


plt.figure(figsize=(8, 5))
plt.grid(linestyle='dotted')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.plot(y.squeeze(-1).detach().numpy(), labels.squeeze(-1).detach().numpy(), 
         label=r'$\sin(-x) + \cos(y)$',
         linestyle="--",
         linewidth=1.8,
         color='red'
        )
plt.plot(y.squeeze(-1).detach().numpy(), y_pred.squeeze(-1).detach().numpy(), 
         label=r'$f(x)$',
         linewidth=1.5,
         color='blue'
        )
plt.legend(ncol=2, loc=2)
plt.tight_layout()
plt.savefig("cosx.svg", dpi=300)

# Plotting the loss
plt.figure(figsize=(8, 5))
plt.grid(linestyle='dotted')
plt.xlabel(r'$epochs$')
plt.ylabel(r'$MSE$')
plt.plot(history, label=r'$MSE$',
         linewidth=1.5,
         color='blue')
plt.tight_layout()
plt.savefig("loss_sincos.svg", dpi=300)