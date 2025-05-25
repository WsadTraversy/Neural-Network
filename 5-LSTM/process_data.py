import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def get_data(set: str):

    with open('data/train.pkl', 'rb') as file:
        data = pickle.load(file)

    (x_data, targets)  = zip(*[(x[0], x[1]) for x in data]) 

    train_split = int(len(x_data) * 0.7)
    train_x_seq = x_data[:train_split]
    test_x_seq = x_data[train_split:]

    all_values = np.concatenate(train_x_seq)
    mean = np.mean(all_values)
    std = np.std(all_values)

    normalized_x_train = [torch.tensor((x - mean) / std).type(torch.float32) for x in train_x_seq]
    normalized_x_test = [torch.tensor((x - mean) / std).type(torch.float32) for x in test_x_seq]
    if set == 'train':
        return (normalized_x_train, targets[:train_split])
    elif set == 'test':
        return (normalized_x_test, targets[train_split:])
    else:
        raise ValueError('Choose beetwen train, test sets')


def pad_collate(batch, pad_value=0):
    xx, yy = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [1]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy = torch.tensor(yy).type(torch.int64) 

    return xx_pad, yy, x_lens, y_lens