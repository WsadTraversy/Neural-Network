from torch.utils.data import Dataset
from process_data import get_data


class VariableDataset(Dataset):
    def __init__(self, set: str):
        (in_data, target) = get_data(set)
        self.data = [(x, y) for x, y in zip(in_data, target)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_data, target = self.data[idx]
        return in_data, target