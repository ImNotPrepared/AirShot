
from re import L
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple


class SCR(nn.Module):
    def __init__(self, channel, res=3, stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
        self.res = res
        super(SCR, self).__init__()
        planes=[640, 64, 64, 64, 640]
        planes[0], planes[-1] = channel, channel
        #planes[1:-1] = 64 * res
        self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[2]),
                                   nn.ReLU())
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]))

    def forward(self, x):
        b, c, h, w, u, v = x.shape # torch.Size([1, 1024, 38, 57, 3, 3])
        x = x.view(b, c, h * w, u * v)
        x = self.conv1x1_in(x)   # representation: [b, 640, hw, 25] -> [b, 64, HW, 25]  spatial size of84×84 
        x = x.view(b, x.shape[1], h * w, u, v)
        x = self.conv1(x)  # [b, 64, hw, 5, 5] --  > [b, 64, hw, 3, 3]  torch.Size([1, 64, 2166, 1, 1]) 
        #x = self.conv2(x)  # [b, 64, hw, 3, 3] --> [b, 64, hw, 1, 1]
        c = x.shape[1]
        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)  # [b, 64, h, w] --> [b, 640, h, w]
        return x

class SelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size, padding=2):
        super(SelfCorrelationComputation, self).__init__()  
        self.kernel_size = (kernel_size, kernel_size)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=3)
        
        identity = x
        x = self.unfold(x)  # b, cuv, h, w          
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w) #  1, c, u, v, h, w
        x = x * identity.unsqueeze(2).unsqueeze(2)  # (1, c, u, v, h, w) * (1, c, 1, 1, h, w)
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # 1, c, h, w, u, v
        return x
    
class RENet(nn.Module):
    
    def __init__(self, channel, res=3):
        super().__init__()

        self.scr_module = self._make_scr_layer(channel, res)
        for modules in [self.scr_module]:
            for l in modules.modules():
                print(l)

    def _make_scr_layer(self, channel, res):
        stride, kernel_size, padding =  (1, 1, 1), -2 * res + 11, 5-res      #(1, 1, 1), 3, 1
        layers = list()
        corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
        self_block = SCR(channel, res=res, ksize=kernel_size, stride=stride)
        layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)

    def forward(self, input):
        org = input
        x = self.scr_module(input) # calculate neigh info
        x = x + org
        x = F.relu(x)
        return x