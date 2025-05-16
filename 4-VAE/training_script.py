import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import VAE
from process_data import TrafficDataset


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

dataset = TrafficDataset()

train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset, batch_size=8, shuffle=True)

vae = VAE(latent_dim=32, hidden_dim=256, x_dim=1024*3).to(device)

criterion = nn.MSELoss(reduction="sum")
optimizer = optim.Adam(vae.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

def vae_loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

num_epochs = 30
for n in range(num_epochs):
    losses_epoch = []
    for x in train_loader:
        x = x.to(device)
        out, means, log_var = vae(x)
        loss = vae_loss_function(x, out, means, log_var)
        losses_epoch.append(loss.item())
        loss.backward()               # backward pass (compute parameter updates)
        optimizer.step()              # make the updates for each parameter
        optimizer.zero_grad()
    L1_list = []
    for x in test_loader:
        x  = x.to(device)
        out, _, _ = vae(x)
        L1_list.append(torch.mean(torch.abs(out-x)).item())
    print(f"Epoch {n} loss {np.mean(np.array(losses_epoch))}, test L1 = {np.mean(L1_list)}")
    scheduler.step()
