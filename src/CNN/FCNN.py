import torch
import torch.nn as nn
import torch.nn.functional as F



class FCNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(FCNN, self).__init__()

        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, out_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)


