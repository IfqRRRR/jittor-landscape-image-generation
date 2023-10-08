import jittor as jt
from jittor import models
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from pretrained.deeplab_jittor.decoder import Decoder
from pretrained.deeplab_jittor.aspp import ASPP
from pretrained.deeplab_jittor.backbone import resnet101

class DeepLab(Module):
    def __init__(self, output_stride=16, num_classes=21):
        super(DeepLab, self).__init__()
        self.backbone = resnet101(output_stride=output_stride)
        self.aspp = ASPP(output_stride)
        self.decoder = Decoder(num_classes)

    def execute(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = nn.resize(x, size=(input.shape[2], input.shape[3]), mode='bilinear')
        
        return x
