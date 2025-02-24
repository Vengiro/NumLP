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
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)


