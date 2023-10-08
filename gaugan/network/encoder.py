from jittor import nn
import jittor as jt
import numpy as np


class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        def get_out_channel(layer):
            if hasattr(layer, 'out_channels'):
                return getattr(layer, 'out_channels')
            return layer.weight.size(0)
    
        def add_norm_layer(layer):
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
            return nn.Sequential(layer, norm_layer)
        
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = add_norm_layer
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2)
        self.opt = opt

    def execute(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = nn.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return eps.mul(std) + mu
    
    def encode_z(self, img):
        mu, logvar = self(img)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar