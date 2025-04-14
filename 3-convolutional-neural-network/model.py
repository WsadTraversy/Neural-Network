import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.bns2 = nn.BatchNorm2d(16)
        self.bns3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(p=0.1)

        self.linear1 = nn.Linear(13456, 8192)
        self.linear2 = nn.Linear(8192, 2048)
        self.linear3 = nn.Linear(2048, 512)
        self.linear4 = nn.Linear(512, 50)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.bns2(x)
        x = self.pool(self.act(self.conv3(x)))
        x = self.bns3(x)
        x = self.act(self.conv4(x))
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.act(self.linear1(x))
        x = self.dropout(x)
        x = self.act(self.linear2(x))
        x = self.dropout(x)
        x = self.act(self.linear3(x))
        x = self.linear4(x)
        return x