"""
Implementations of different CNNs

by Christopher M. Sandino (sandino@stanford.edu), 2019.

"""

import torch
from torch import nn
from MoDLsinglechannel.demo_modl_BeaChloe.utils.transforms_modl import center_crop


class ConvBlock(nn.Module):
    """
    A 2D Convolutional Block that consists of Norm -> ReLU -> Dropout -> Conv

    Based on implementation described by: 
        K He, et al. "Identity Mappings in Deep Residual Networks" arXiv:1603.05027
    """
    def __init__(self, in_chans, out_chans, kernel_size, drop_prob, conv_type='conv2d',
                 act_type='relu', norm_type='none'):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        padding = 1 if kernel_size>1 else 0 

        # Define choices for each layer in ConvBlock
        normalizations = nn.ModuleDict([
                ['none',     nn.Identity()],
                ['instance', nn.InstanceNorm2d(in_chans, affine=False)],
                ['batch',    nn.BatchNorm2d(in_chans, affine=False)]
        ])
        activations = nn.ModuleDict([
                ['relu',  nn.ReLU()],
                ['leaky_relu', nn.LeakyReLU()]
        ])
        dropout = nn.Dropout2d(p=drop_prob)
        convolution = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding)

        # Define forward pass
        self.layers = nn.Sequential(
            normalizations[norm_type], 
            activations[act_type],
            dropout, convolution
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock2D(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class ResBlock(nn.Module):
    """
    A ResNet block that consists of two convolutional layers followed by a residual connection.
    """

    def __init__(self, in_chans, out_chans, kernel_size, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_chans, out_chans, kernel_size, drop_prob),
            ConvBlock(out_chans, out_chans, kernel_size, drop_prob)
        )

        if in_chans != out_chans:
            self.resample = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.resample = nn.Identity()

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """

        # To have a residual connection, number of inputs must be equal to outputs
        shortcut = self.resample(input)

        return self.layers(input) + shortcut


class ResNet(nn.Module):
    """
    Prototype for 3D ResNet architecture
    """

    def __init__(self, num_resblocks, in_chans, chans, kernel_size, drop_prob, circular_pad=True):
        """

        """
        super().__init__()

        self.circular_pad = circular_pad
        self.pad_size = 2*num_resblocks + 1

        # Declare ResBlock layers
        self.res_blocks = nn.ModuleList([ResBlock(in_chans, chans, kernel_size, drop_prob)])
        for _ in range(num_resblocks-1):
            self.res_blocks += [ResBlock(chans, chans, kernel_size, drop_prob)]

        # Declare final conv layer (down-sample to original in_chans)
        self.final_layer = nn.Conv2d(chans, in_chans, kernel_size=kernel_size, padding=1)


    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.in_chans, depth, width, height]
        """

        orig_shape = input.shape
        if self.circular_pad:
            input = nn.functional.pad(input, 2*(self.pad_size,) + (0,0), mode='circular')

        # Perform forward pass through the network
        output = input
        for res_block in self.res_blocks:
            output = res_block(output)
        output = self.final_layer(output) + input

        return center_crop(output, orig_shape)

