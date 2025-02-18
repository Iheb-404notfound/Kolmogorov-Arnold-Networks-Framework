from src.kan import KAN_Layer
import torch
import torch.nn as nn
from torch import optim
from torch.functional import F
import matplotlib.pyplot as plt

knot_number = 10
Bsplines = lambda x, mu = 0, s = 1: torch.exp(-(x-mu)**2/(2*s**2))/(s*torch.math.sqrt(2*torch.pi))
kan = KAN_Layer(1, 1, knot_number, Bsplines)

x = torch.linspace(1,3*torch.pi,100).unsqueeze(-1)
# y_true = torch.sin(torch.linspace(0,3*torch.pi,100)).unsqueeze(-1)
y_true = torch.log(torch.linspace(1,3*torch.pi,100)).unsqueeze(-1)
optimizer = optim.Adam(kan.parameters(), lr=0.001)
history = []

# Training Loop
print("Training Loop")
for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = kan(x)
    loss = nn.MSELoss()(y_pred, y_true)
    history.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}", end='\r')
    loss.backward()
    optimizer.step()


# Plotting the results
plt.rcParams.update({'font.size': 16})
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(8, 5))
plt.grid(linestyle='dotted')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.plot(x.squeeze(-1).detach().numpy(), y_true.squeeze(-1).detach().numpy(), 
         label=r'$\log{x}$',
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
plt.savefig("log.svg", dpi=300)

# Plotting the loss
# plot history
plt.figure(figsize=(8, 5))
plt.grid(linestyle='dotted')
plt.xlabel(r'$epochs$')
plt.ylabel(r'$MSE$')
plt.plot(history, label=r'$MSE$',
         linewidth=1.5,
         color='blue')
plt.tight_layout()
plt.savefig("loss.svg", dpi=300)