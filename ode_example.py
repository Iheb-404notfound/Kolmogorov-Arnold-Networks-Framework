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
knots = 10

def generator_function(x, mu = 0, s = 1):
    mu = mu * 4/knots
    return torch.exp(-(x-mu)**2/(2*s**2))/(s*torch.math.sqrt(2*torch.pi))
generator_function = lambda x, mu = 0, s = 1: torch.exp(-(x-mu)**2/(2*s**2))/(s*torch.math.sqrt(2*torch.pi))

features = torch.stack([torch.linspace(-2, 2, 100)], dim=-1)

# Define the ODE and boundary condition
def loss_function(model, x, y0):
    x.requires_grad = True  # Enable gradient tracking for x
    y = model(x)            # Neural network prediction
    dydx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]  # Compute dy/dx
    residual = dydx - 2 * y  # ODE residual
    
    # Boundary condition at x=0
    bc_loss = (model(torch.tensor([[0.0]])) - y0) ** 2
    
    # Total loss: physics + boundary condition
    return torch.mean(residual**2) + bc_loss

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KanNetwork(input_size, hidden_size, output_size, knots, generator_function).to(device)

optimizer = optim.Adam(model.parameters())
num_epochs = 30_000
history = []

# Training data
x = features.to(device)  # x in [0, 2]
y0 = torch.tensor([[1.0]], device=device)  # Initial condition y(0) = 1

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_function(model, x, y0)
    loss.backward()
    optimizer.step()
    history.append(loss.item())
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    #if epoch % 2000 == 0:
    #    # Test the trained model
    #    x_test = features.to(device)
    #    y_pred = model(x_test).detach().cpu()
#
    #    x_test = x_test.squeeze(-1)
    #    y_pred = y_pred.squeeze(-1)
    #    y_exact = y0.item() * torch.exp(2 * x_test).cpu()  # Exact solution
#
    #    plt.plot(x_test.detach().numpy(), y_pred.detach().numpy(), label="NN Solution")
    #    plt.plot(x_test.detach().numpy(), y_exact.detach().numpy(), label="Exact Solution", linestyle='dashed')
    #    plt.xlabel("x")
    #    plt.ylabel("y")
    #    plt.legend()
    #    plt.show()

# Plotting the results
x_test = features.to(device)
y_pred = model(x_test).detach().cpu()
x_test = x_test.squeeze(-1)
y_pred = y_pred.squeeze(-1)
y_exact = y0.item() * torch.exp(2 * x_test).cpu()  # Exact solution
plt.rcParams.update({'font.size': 16})
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(8, 5))
plt.grid(linestyle='dotted')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.plot(features.squeeze(-1).detach().numpy(), y_exact.squeeze(-1).detach().numpy(), 
         label=r'$e^{2x}$',
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
plt.savefig("ode.svg", dpi=300)

# Plotting the loss
plt.figure(figsize=(8, 5))
plt.grid(linestyle='dotted')
plt.xlabel(r'$epochs$')
plt.ylabel(r'$eq$')
plt.plot(history, label=r'$eq$',
         linewidth=1.5,
         color='blue')
plt.tight_layout()
plt.savefig("loss_ode.svg", dpi=300)