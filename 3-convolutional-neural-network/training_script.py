from model import ConvNet
from process_data import get_data
import torch
import numpy as np
from torch import nn, optim



device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

trainloader, valloader = get_data()

def accuracy(outputs, targets):
    correct = 0
    total = 0
    _, preds = torch.max(outputs, 1)
    correct += (preds == targets).sum().item()
    total += targets.size(0)

    return correct/total

train_losses = []
valid_losses = []
train_acc = []
valid_acc = []
best_valid_loss = np.inf
best_valid_acc = 0
for epoch in range(20):
    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_train_acc = []
    epoch_valid_acc = []
    for data in iter(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        model.train()

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        epoch_train_losses.append(loss.item())
        epoch_train_acc.append(accuracy(outputs, labels))

        optimizer.step()
    
    for data in iter(valloader):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            model.eval()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            epoch_valid_losses.append(loss.item())
            epoch_valid_acc.append(accuracy(outputs, labels))
    
    train_loss = np.array(epoch_train_losses).mean()
    valid_loss = np.array(epoch_valid_losses).mean()
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_acc.append(np.mean(epoch_train_acc))
    valid_acc.append(np.mean(epoch_valid_acc))
    print(f"Epoch {epoch+1} train_loss {train_loss:.3} train_acc {train_acc[-1]:.3f} valid_loss {valid_loss:.3f} valid_acc: {valid_acc[-1]:.3}")

    if valid_acc[-1] > best_valid_acc:
        best_valid_loss = valid_loss
        best_valid_acc = valid_acc[-1]
        best_weights = model.state_dict()

print(f"Best validation loss: {best_valid_loss:.4f}, Best validation accuracy: {best_valid_acc:.4f}")

model.load_state_dict(best_weights)
torch.save(model.state_dict(), "state_dict.pickle")
        

