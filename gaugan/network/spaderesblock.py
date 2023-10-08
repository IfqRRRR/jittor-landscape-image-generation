"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from jittor import nn
import jittor as jt
# import torch.nn.utils.spectral_norm as spectral_norm
from network.spade import SPADE
# from utils.SpectralNorm import SpectralNorm

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):     #fin: x的通道数  fout:目标通道数
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        # if 'spectral' in opt.norm_G:
        #     self.conv_0 = spectral_norm(self.conv_0)
        #     self.conv_1 = spectral_norm(self.conv_1)
        #     if self.learned_shortcut:
        #         self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        # spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(fin, opt)     # semanctic_nc
        self.norm_1 = SPADE(fmiddle, opt)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, opt)

    def execute(self, x, seg):  # (batch_size, 1024, 12, 16) (batch_size, 3, 384, 512)
        x_s = self.shortcut(x, seg) # (batch_size, fout, 12, 16)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))  
        # norm_0(x, seg): (batch_size, 1024, 12, 16) conv_0(..): (batch_size, fout, 12, 16)
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))  # (batch_size, fout, 12, 16)

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
            # norm_s(x, seg): (batch_size, 1024, 12, 16) conv_s(..): (batch_size, fout, 12, 16)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return nn.leaky_relu(x, 2e-1)


