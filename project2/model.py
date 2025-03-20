import torch
import torch.nn as nn


class PriceClassifier(nn.Module):
    def __init__(self, numerical_shape=12, categorical_shape=7):
        super(PriceClassifier, self).__init__()
        self.emb_layer = nn.Linear(categorical_shape, categorical_shape)
        self.emb_act = nn.Sigmoid()
        self.input = nn.Linear(numerical_shape+categorical_shape, 128)
        self.act = nn.ReLU()
        self.output = nn.Linear(128, 3)

    def forward(self, x, cat_x):
        cat_x_embedded = self.emb_layer(cat_x)
        cat_x_embedded = self.emb_act(cat_x_embedded)
        x = torch.cat([x, cat_x_embedded], dim=1)
        x = self.act(self.input(x))
        x = self.output(x)
        return x
