# CMU 18-780/6 Homework 4
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
# For the assignment, you are asked to create the architectures of these
# three networks by filling in the __init__ and forward methods in the
# DCGenerator, DCDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn

def spectral_norm(S, num_iters=1):

    if isinstance(S, nn.Sequential):
        W = S[0].weight
    W_shape = W.shape
    W = W.view(W_shape[0], -1)  # Flatten the weight matrix if it's higher dimensional

    u = torch.randn(W.shape[0], device=W.device)
    v = torch.randn(W.shape[1], device=W.device)

    for i in range(num_iters):
        v = torch.nn.functional.normalize(torch.matmul(W.T, u), dim=0)
        u = torch.nn.functional.normalize(torch.matmul(W, v), dim=0)

    sigma = torch.dot(u, torch.matmul(W, v))
    W_sn = W / sigma
    S[0].weight.data = W_sn.view(W_shape)

def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    elif activ == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif activ == 'none':
        pass
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):

    def __init__(self, noise_size, conv_dim=64):
        super().__init__()

        self.up_conv1 = conv(noise_size, conv_dim*8, 4, 1, 3, None, False, 'leaky')
        self.up_conv2 = up_conv(conv_dim*8, conv_dim*4, 4, 2, 1, 4, 'instance', 'leaky')
        self.up_conv3 = up_conv(conv_dim*4, conv_dim*2, 4, 2, 1, 4, 'instance', 'leaky')
        self.up_conv4 = up_conv(conv_dim*2, conv_dim, 4, 2, 1, 4, 'instance', 'leaky')
        self.up_conv5 = up_conv(conv_dim, 3, 4, 2, 1, 4, None, 'tanh')

    def forward(self, z):
        """
        Generate an image given a sample of random noise.

        Input
        -----
            z: BS x noise_size x 1 x 1   -->  16x100x1x1

        Output
        ------
            out: BS x channels x image_width x image_height  -->  16x3x64x64
        """


        out = self.up_conv1(z)
        out = self.up_conv2(out)
        out = self.up_conv3(out)
        out = self.up_conv4(out)
        out = self.up_conv5(out)
        return out


class ResnetBlock(nn.Module):

    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm,
            activ=activ
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out



class DCDiscriminatorvar(nn.Module):
    """Architecture of the discriminator network."""

    def __init__(self, conv_dim=64, norm='instance'):
        super().__init__()
        self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'leaky')
        self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'leaky')
        self.conv3 = conv(64, 128, 4, 2, 1, norm, False, 'leaky')
        self.conv4 = conv(128, 256, 4, 2, 1, norm, False, 'leaky')
        # Use Sigmoid because BCELoss is used so the output should be in [0, 1]
        self.conv5 = conv(256, 1, 4, 1, 0, None, activ='sigmoid')

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        spectral_norm(self.conv1)
        x = self.conv1(x)
        spectral_norm(self.conv2)
        x = self.conv2(x)
        spectral_norm(self.conv3)
        x = self.conv3(x)
        spectral_norm(self.conv4)
        x = self.conv4(x)
        spectral_norm(self.conv5)
        x = self.conv5(x)
        return x.squeeze()

class Critic(nn.Module):

    def __init__(self, conv_dim=64, norm='instance'):
        super().__init__()
        self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'leaky')
        self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'leaky')
        self.conv3 = conv(64, 128, 4, 2, 1, norm, False, 'leaky')
        self.conv4 = conv(128, 256, 4, 2, 1, norm, False, 'leaky')
        self.conv5 = conv(256, 1, 4, 1, 3, norm=None, activ='none')

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()