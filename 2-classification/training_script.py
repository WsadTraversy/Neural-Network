import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import numpy as np
from process_data import get_data
from model import PriceClassifier


train_dataset, validation_dataset = get_data()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)


device = torch.device("cpu")
model = PriceClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
accuracy = Accuracy(task="multiclass", num_classes=3)

class_weights = torch.tensor([1.38, 7.34, 7.24])
criterion = nn.CrossEntropyLoss(weight=class_weights)


iters = []
train_losses = []
valid_losses = []
train_acc = []
valid_acc = []
best_valid_loss = np.inf
best_valid_acc = 0
for n in range(250):
    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_train_acc = []
    epoch_valid_acc = []
    for x, cat_x, targets in iter(train_loader):
        x, cat_x, targets = x.to(device), cat_x.to(device), targets.to(device)
        model.train()
        outputs = model(x, cat_x).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        epoch_train_losses.append(loss.item())
        epoch_train_acc.append(accuracy(outputs, targets))
        optimizer.step()
        optimizer.zero_grad()
    
    for x, cat_x, targets in iter(validation_loader):
        with torch.no_grad():
            x, cat_x, targets = x.to(device), cat_x.to(device), targets.to(device)
            model.eval()
            outputs = model(x, cat_x).squeeze()
            loss = criterion(outputs, targets)
            epoch_valid_losses.append(loss.item())
            epoch_valid_acc.append(accuracy(outputs, targets))

    train_loss = np.array(epoch_train_losses).mean()
    valid_loss = np.array(epoch_valid_losses).mean()
    iters.append(n)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_acc.append(np.mean(epoch_train_acc))
    valid_acc.append(np.mean(epoch_valid_acc))
    print(f"Epoch {n} train_loss {train_loss:.3} train_acc {train_acc[-1]:.3f} valid_loss {valid_loss:.3f} valid_acc: {valid_acc[-1]:.3}")

    if valid_acc[-1] > best_valid_acc:
        best_valid_loss = valid_loss
        best_valid_acc = valid_acc[-1]
        best_weights = model.state_dict()

print(f"Best validation loss: {best_valid_loss:.4f}, Best validation accuracy: {best_valid_acc:.4f}")

# model.load_state_dict(best_weights)
# torch.save(model.state_dict(), "data/state_dict.pickle")
