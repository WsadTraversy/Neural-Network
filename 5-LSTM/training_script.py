import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import VariableDataset
from process_data import pad_collate
from model import LSTMRegressor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 50

train_dataset = VariableDataset('train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

# test_dataset = VariableDataset('train')
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=pad_collate)

# Zmieniaj 2 i 3 wartość w modelu też bidirectional=True powinno poprawić acc
model = LSTMRegressor(1, 200, 2, 5, bidirectional=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.CrossEntropyLoss()

# Training loop
acc = []
model.train()
for epoch in range(101):
    for x, targets, x_len, target_len in train_loader:
        x = x.to(device).unsqueeze(2)
        targets = targets.to(device)
        # x_packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        hidden, state = model.init_hidden(x.size(0))
        hidden, state = hidden.to(device), state.to(device)
        preds, _ = model(x, (hidden, state))
        preds = preds.squeeze(1)
        optimizer.zero_grad()
        loss = loss_fun(preds, targets)
        loss.backward()
        optimizer.step()
        acc.append((torch.argmax(preds, dim=1) == targets).sum().item()/BATCH_SIZE)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}, accuracy: {np.mean(acc):.3}")
        acc = []

# with torch.no_grad():
#     for x, targets, x_len, target_len in test_loader:
#         x = x.to(device).unsqueeze(2)
#         targets = targets.to(device)
#         hidden, state = model.init_hidden(x.shape[0])
#         hidden, state = hidden.to(device), state.to(device)

#         x_packed = pack_padded_sequence(
#             x, x_len, batch_first=True, enforce_sorted=False
#         )
#         preds_packed, _ = model(x_packed, (hidden, state))
#         preds, pred_len = pad_packed_sequence(
#             preds_packed, batch_first=True, padding_value=0
#         )

#         preds = preds.squeeze(2)
#         mask_tgt = targets != 0