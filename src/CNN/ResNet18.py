import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down=False):
        super(ResBlock, self).__init__()
        stride = 2 if down else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)

        self.skip = nn.Identity() if not down \
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x += skip
        return self.relu(x)

    def __call__(self,x):
        return self.forward(x)

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        #7x7 Conv is too big for CIFAR, so we use 3x3 Conv
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.res1 = ResBlock(64, 64, 3)
        self.res2 = ResBlock(64, 64, 3)
        self.res3 = ResBlock(64, 128, 3, down=True)
        self.res4 = ResBlock(128, 128, 3)
        self.res5 = ResBlock(128, 256, 3, down=True)
        self.res6 = ResBlock(256, 256, 3)
        self.res7 = ResBlock(256, 512, 3, down=True)
        self.res8 = ResBlock(512, 512, 3)
        self.fc = nn.Linear(512*4*4, 100)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc(x)
        return x

    def __call__(self, x):
        return self.forward(x)
