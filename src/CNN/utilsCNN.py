import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy as sp

class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias



        # He initialization for the weights
        # Parameters otherwise they don't get optimized
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)*np.sqrt(2/(in_channels*kernel_size*kernel_size)))
        # it gets broadcasted to (out_channels, in_channels, kernel_size, kernel_size)
        self.b = nn.Parameter(torch.zeros(out_channels))

            
    
    def __call__(self, x):
        
        return self.forward(x)


    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """


        out_height = (x.shape[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (x.shape[3] - self.kernel_size + 2 * self.padding) // self.stride + 1

        if self.padding > 0:
            # mode = 'constant' with zero padding
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        # For loops way too slow

        # (Batch, out_channels*kernel_size*kernel_size, kernel_positions)
        flatX = F.unfold(x, self.kernel_size, stride=self.stride)
        # (out_channels, in_channels*kernel_size*kernel_size)
        flatW = self.W.view(self.out_channels, -1)
        # At the end of the day these convolutions are just matrix multiplications
        flatOut = torch.matmul(flatW, flatX)
        # Reshape to (Batch, out_channels, out_height, out_width)
        output = flatOut.view(x.shape[0], self.out_channels, out_height, out_width) + self.b.view(1, self.out_channels, 1, 1)


        return output
    
class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size


        self.stride = stride if stride is not None else kernel_size




    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """
        
        ## check the dimensions
        self.batch_size = x.shape[0]
        self.channel = x.shape[1]
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]
        

        self.output_height   = (self.input_height - self.kernel_size) // self.stride + 1
        self.output_width    = (self.input_width - self.kernel_size) // self.stride + 1
        self.output_channels = self.channel
        self.x_pool_out      = None

        # first unfold create vertical window (batch, channel, windowV, kernel_size, output_width)
        # second unfold create horizontal window for each vertical window
        # (batch, channel, windowV, windowH, kernel_size, kernel_size)
        # We need to find the max value in each window
        flatX = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # collapse kernel_size
        flatX = flatX.reshape(self.batch_size, self.channel, flatX.shape[2], flatX.shape[3], -1)
        # find max value in each window
        self.x_pool_out = flatX.max(-1)[0]


        return self.x_pool_out

"""
    Test using the given image. You need it in the same directory as this script.
    Test1: Take the image and apply the convolution layer with the given kernel size, stride, and padding.
    we can
"""
def main():
    data = torch.randn(1, 3, 10, 10)
    test1Conv(data, 10)
    test1Pool(data, 10)
    test2Conv(1, 3, 1, 1)
    test2Pool(2, 2)


def test1Conv(data, size):
    l1 = MyConv2D(3, 32, 3, 2, 1)
    res1 = l1(data)
    assert res1.shape == (1, 32, size//2, size//2)
    print('Test 1 convolution passed.')

def test2Conv(out_channels, kernel_size, stride, padding):
    img = Image.open('2007_001239.jpg')
    h, w = img.size
    img = np.array(img)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    l1 = MyConv2D(3, out_channels, kernel_size, stride, padding)
    res1 = l1(img)
    new_h = (h + 2*padding - kernel_size) // stride + 1
    new_w = (w + 2*padding - kernel_size) // stride + 1
    assert res1.shape == (1, out_channels, new_w, new_h)
    print('Test 2 convolution passed.')
    Image.fromarray(res1.squeeze(0).squeeze(0).byte().numpy()).show()

def test1Pool(data, size):
    l2 = MyMaxPool2D(2, 2)
    res2 = l2(data)
    assert res2.shape == (1, 3, size//2, size//2)
    print('Test 1 maxPooling passed.')

def test2Pool(kernel_size, stride=None):
    img = Image.open('2007_001239.jpg')
    h, w = img.size
    img = np.array(img)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    l2 = MyMaxPool2D(kernel_size, stride)
    res2 = l2(img)
    new_h = (h - kernel_size) // stride + 1
    new_w = (w - kernel_size) // stride + 1
    assert res2.shape == (1, 3, new_w, new_h)
    print('Test 2 maxPooling passed.')
    Image.fromarray(res2.squeeze(0).permute(1, 2, 0).byte().numpy()).show()


if __name__ == "__main__":

    ## Test your implementation!
    main()


