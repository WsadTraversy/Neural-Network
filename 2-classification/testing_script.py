import torch
import csv
from process_data import get_data 
from model import PriceClassifier

test_dataset = get_data(test_dataset=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

device = torch.device("cpu")
model = PriceClassifier().to(device)
model.load_state_dict(torch.load('data/state_dict.pickle'))

for x, cat_x in iter(test_loader):
    with torch.no_grad():
        x, cat_x = x.to(device), cat_x.to(device)
        model.eval()
        outputs = model(x, cat_x).squeeze()

list_to_save = []

# average, cheap, expensive
for el in outputs:
    _, index = torch.max(el, 0)
    if index == 0:
        list_to_save.append(1)
    elif index == 1:
        list_to_save.append(0)
    elif index == 2:
        list_to_save.append(2)
    else:
        raise(ValueError('Wrong indexes'))

with open('data/koncowy.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    for item in list_to_save:
        writer.writerow([item])

print("File saved!")
