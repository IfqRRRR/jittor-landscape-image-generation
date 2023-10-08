from jittor import nn
import jittor as jt

class SPADE(nn.Module):
    def __init__(self, norm_nc, opt):
        super().__init__()

        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        # ks = int(parsed.group(2))

        # if param_free_norm_type == 'instance':
        #     self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'batch':
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        ks = 3
        # else:
        #     raise ValueError('%s is not a recognized param-free norm type in SPADE'
        #                      % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        seg_chanel = opt.label_c
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(seg_chanel, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def execute(self, x, segmap):   # (batch_size, 1024, 12, 16) (batch_size, 3, 384, 512)
        normalized = self.param_free_norm(x)    # (batch_size, 1024, 12, 16)
        segmap = nn.interpolate(segmap, size=x.size()[2:], mode='nearest')   
        # size=x.size()[2:]: [12, 16] segmap: (batch_size, 3, 12, 16)
        actv = self.mlp_shared(segmap)  # (batch_size, 128, 12, 16)
        gamma = self.mlp_gamma(actv)    # (batch_size, norm_nc, 12, 16)
        beta = self.mlp_beta(actv)      # (batch_size, norm_nc, 12, 16)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out