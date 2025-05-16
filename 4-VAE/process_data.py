from torch.utils.data import Dataset
from glob import glob
import os
from torchvision import transforms
from PIL import Image



class TrafficDataset(Dataset):
    def __init__(self):
        super().__init__()
        self._paths = sorted(glob(os.path.join('data/trafic_32/', '*', '*', '*.jpg')))
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, index):
        img_path = self._paths[index]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Błąd przy otwieraniu {img_path}: {e}")
            image = Image.new("RGB", (32, 32), (0, 0, 0))
        return self.transform(image)

        
