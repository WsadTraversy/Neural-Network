import pickle
import numpy as np
import torch
import random
from torch.nn.utils.rnn import pad_sequence

def get_data(set: str):

    with open('data/train.pkl', 'rb') as file:
        data = pickle.load(file)

    (x_data, targets)  = zip(*[(x[0], x[1]) for x in data]) 

    all_values = np.concatenate(x_data)
    mean = np.mean(all_values)
    std = np.std(all_values)

    normalized_x = [torch.tensor((x - mean) / std).type(torch.float32) for x in x_data]
    normalized_data = list(zip(normalized_x, targets))
    
    data_0 = [x for x in normalized_data if x[1] == 0]
    random.shuffle(data_0)
    data_1 = [x for x in normalized_data if x[1] == 1]
    random.shuffle(data_1)
    data_2 = [x for x in normalized_data if x[1] == 2]
    random.shuffle(data_2)
    data_3 = [x for x in normalized_data if x[1] == 3]
    random.shuffle(data_3)
    data_4 = [x for x in normalized_data if x[1] == 4]
    random.shuffle(data_4)

    data_train = random.sample(data_0, k=500) + random.choices(data_1, k=500) + random.choices(data_2, k=500) + random.choices(data_3, k=500) + random.choices(data_4, k=500)
    (x_train, targets) = zip(*[(x[0], x[1]) for x in data_train])
    if set == 'train':
        return (x_train, targets)
    elif set == 'test':
        return None
    else:
        raise ValueError('Choose beetwen train, test sets')


def pad_collate(batch, pad_value=0):
    xx, yy = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [1]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy = torch.tensor(yy).type(torch.int64) 

    return xx_pad, yy, x_lens, y_lens