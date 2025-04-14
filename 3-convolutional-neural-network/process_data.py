import random
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_data():
    transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    augmentation = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.RandomHorizontalFlip(0.5),
     transforms.RandomVerticalFlip(0.5),
     transforms.RandomRotation(24),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = ImageFolder("data/", transform=transform)
    augset = ImageFolder("data/", transform=augmentation)
    valset = ImageFolder("data/", transform=transform)

    # Etykieta 10 - 1108 obrazów; Etykieta 13 - 503 obrazy; Reszta etykiet - 1800 obrazów
    # 50 x 50 = 2500

    # ilosc obrazow + ilosc obrazow po augmentacji dla danej klasy w dataset -> (64+64)x50
    train_num_of_images = 64
    val_num_of_images = 32

    train_set_limit = {}
    aug_set_limit = {}
    val_set_limit = {}
    lower_limit = 0
    higher_limit = 1800
    for i in range(50):
        if i == 10:
            train_set_limit[i] = random.sample(range(lower_limit, higher_limit), train_num_of_images)
            aug_set_limit[i] = random.sample(range(lower_limit, higher_limit), train_num_of_images)
            val_set_limit[i] = random.sample(range(lower_limit, higher_limit), val_num_of_images)
        elif i == 13:
            train_set_limit[i] = random.sample(range(lower_limit, higher_limit), train_num_of_images)
            aug_set_limit[i] = random.sample(range(lower_limit, higher_limit), train_num_of_images)
            val_set_limit[i] = random.sample(range(lower_limit, higher_limit), val_num_of_images)
        else:
            train_set_limit[i] = random.sample(range(lower_limit, higher_limit), train_num_of_images)
            aug_set_limit[i] = random.sample(range(lower_limit, higher_limit), train_num_of_images)
            val_set_limit[i] = random.sample(range(lower_limit, higher_limit), val_num_of_images)
        
        if i == 9:
            lower_limit += 1800
            higher_limit += 1108
        elif i == 10:
            lower_limit += 1108
            higher_limit += 1800
        elif i == 12:
            lower_limit += 1800
            higher_limit += 503
        elif i == 13:
            lower_limit += 503
            higher_limit += 1800
        else:
            lower_limit += 1800
            higher_limit += 1800

    trainset.samples = [(path, label) for index, (path, label) in enumerate(trainset.samples) if index in train_set_limit[label]]
    augset.samples = [(path, label) for index, (path, label) in enumerate(augset.samples) if index in aug_set_limit[label]]
    valset.samples = [(path, label) for index, (path, label) in enumerate(valset.samples) if index in val_set_limit[label]]
    
    trainset = trainset + augset

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    val_loader = DataLoader(valset, batch_size=32, shuffle=False)

    return train_loader, val_loader
