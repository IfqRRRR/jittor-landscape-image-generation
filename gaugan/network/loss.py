import jittor.nn as nn
import jittor as jt
from jittor.models import vgg19


class Ganloss(nn.Module):
    def __init__(self, mode):
        super(Ganloss, self).__init__()
        self.ganloss = nn.BCEWithLogitsLoss()
        self.mode = mode
        self.zero_tensor = None
        
    def __call__(self, input, target):
        loss = 0
        for each_D in input:
            each_D = each_D[-1]
            loss += self.ganloss(each_D, target) 
            # if self.mode == 'd':
            #     if target:
            #         minval = jt.minimum(each_D - 1, jt.zeros_like(each_D))
            #         loss += -jt.mean(minval)
            #     else:
            #         minval = jt.minimum(-each_D - 1, jt.zeros_like(each_D))
            #         loss += -jt.mean(minval)
            # elif self.mode == 'g':
            #     loss += -jt.mean(each_D)
        return loss / len(input)
    
    # def get_zero_tensor(self, input):
    #     if self.zero_tensor is None:
    #         self.zero_tensor = jt.float(0)
    #         self.zero_tensor.stop_grad()
    #     return self.zero_tensor.expand_as(input)

class KLDLoss(nn.Module):
    def execute(self, mu, logvar):
        return -0.5 * jt.sum(1 + logvar - mu.pow(2) - logvar.exp())  

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

#for perceptual loss
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def execute(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
    
class KLDLoss(nn.Module):
    def execute(self, mu, logvar):
        return -0.5 * jt.sum(1 + logvar - mu.pow(2) - logvar.exp())