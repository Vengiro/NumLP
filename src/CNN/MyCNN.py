import torch
import torch.nn as nn
import torch.nn.functional as F
from utilsCNN import *

class CNN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()

        self.conv1 = MyConv2D(in_channels, 32, 3, 2, 1)
        self.conv2 = MyConv2D(32, 64, 3, 2, 1)
        self.conv3 = MyConv2D(64, 128, 3, 2, 1)
        self.fc1 = nn.Linear(128*4*4, out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)