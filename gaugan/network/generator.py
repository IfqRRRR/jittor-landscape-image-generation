"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from jittor import nn
import jittor as jt
# from spaderesblock import ResnetBlock as ResnetBlock
from network.spaderesblock import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.nf = opt.ngf    # 64

        # self.sw, self.sh = self.compute_latent_vector_size(opt)
        self.sw = 512 // 128  # 16
        self.sh = 384 // 128  # 12
        self.z_dim = 256 
        # if opt.use_vae:
            # In case of VAE, we will sample from random z vector
        self.fc = nn.Linear(self.z_dim, 16 * self.nf * self.sw * self.sh)
        # self.fc = nn.Conv2d(3, 16 * self.nf, 3, padding=1)
        # else:
        #     # Otherwise, we make the network deterministic by starting with
        #     # downsampled segmentation map instead of random z
        #     self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * self.nf, 16 * self.nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * self.nf, 16 * self.nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * self.nf, 16 * self.nf, opt)

        self.up_0 = SPADEResnetBlock(16 * self.nf, 8 * self.nf, opt)
        self.up_1 = SPADEResnetBlock(8 * self.nf, 4 * self.nf, opt)
        self.up_2 = SPADEResnetBlock(4 * self.nf, 2 * self.nf, opt)
        self.up_3 = SPADEResnetBlock(2 * self.nf, 1 * self.nf, opt)

        final_nc = self.nf

        # if opt.num_upsampling_layers == 'most':
        self.up_4 = SPADEResnetBlock(1 * self.nf, self.nf // 2, opt)
        final_nc = self.nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def execute(self, input, z=None):
        seg = input

        # if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
        if z is None:
            z = jt.randn(input.size(0), self.z_dim)     # (batch_size, 256)
        # x = nn.interpolate(seg, size=(3, 4)) # (batch_size, 3, 3, 4)
        
        x = self.fc(z)                              # (batch_size, 1024 * 4 * 3)
        x = x.view(-1, 16 * self.nf, self.sh, self.sw)  # (batch_size, 1024, 3, 4)

        x = self.head_0(x, seg)         # (batch_size, 1024, 3, 4)

        x = self.up(x)              # (batch_size, 1024, 6, 8)
        x = self.G_middle_0(x, seg)     # (batch_size, 1024, 6, 8)

        # if self.opt.num_upsampling_layers == 'more' or \
        #    self.opt.num_upsampling_layers == 'most':
        x = self.up(x)                  # (batch_size, 1024, 12, 16)

        x = self.G_middle_1(x, seg)     # (batch_size, 1024, 12, 16)

        x = self.up(x)      # (batch_size, 1024, 24, 32)
        x = self.up_0(x, seg)   # (batch_size, 512, 24, 32)
        x = self.up(x)          # (batch_size, 512, 48, 64)
        x = self.up_1(x, seg)   # (batch_size, 256, 48, 64)
        x = self.up(x)          # (batch_size, 256, 96, 128)
        x = self.up_2(x, seg)   # (batch_size, 128, 96, 128)
        x = self.up(x)          # (batch_size, 128, 192, 256)
        x = self.up_3(x, seg)   # (batch_size, 64, 192, 256)

        # if self.opt.num_upsampling_layers == 'most':
        x = self.up(x)          # (batch_size, 64, 384, 512)
        x = self.up_4(x, seg)   # (batch_size, 32, 384, 512)

        x = self.conv_img(nn.leaky_relu(x, 2e-1))   # (batch_size, 3, 384, 512)
        x = jt.tanh(x)  # (batch_size, 3, 384, 512)

        return x