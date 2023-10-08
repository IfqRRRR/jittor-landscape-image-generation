
import jittor as jt
from jittor import nn
# from utils.SpectralNorm import SpectralNorm

def start_grad(model):
    for param in model.parameters():
        if 'running_mean' in param.name() or 'running_var' in param.name(): continue
        param.start_grad()

def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)


class NLayerDiscriminator(nn.Module):

    def __init__(self, opt):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 2 #int(np.ceil((kw - 1.0) / 2))
        nf = 64
        input_nc = 3 + opt.label_c
        layer_num = 4
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2)]]

        for n in range(1, layer_num):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == layer_num - 1 else 2
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw), 
                          nn.InstanceNorm2d(nf),
                        #   nn.BatchNorm2d(nf),
                          nn.LeakyReLU(0.2)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        self.model = nn.Sequential()
        for n in range(len(sequence)):
            self.model.append(nn.Sequential(*sequence[n]))

    def execute(self, input):
        results = [input]
        for submodel in self.model:
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        return results[1:]

class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        num_D = 2
        self.model = nn.Sequential()
        for _ in range(num_D):
            self.model.append(NLayerDiscriminator(opt))
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def execute(self, input):
        result = []
        for D in self.model:
            out = D(input)
            result.append(out)
            input = self.avgpool(input)
        return result