from re import X
import torch.nn as nn
import torch.nn.functional as F
import torch as t
import torch
import time
from math import *
import logging



__all__ = ["Relation_Net"]
logger = logging.getLogger(__name__)

    
    
class Relation_Net(nn.Module):
    
    def __init__(self, in_channels, bot_layer_channel):
        super(Relation_Net, self).__init__()
        #layer definition
        #self.conv_hidden = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #                                nn.BatchNorm2d(bot_layer_channel),
        #                                nn.MaxPool2d(kernel_size=3, stride=1, padding=2),
        #                                nn.ReLU(inplace=True)
        #                                )
        
        self.select = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.mlp_2=nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2),
                nn.ReLU(inplace=True),
                )
                #layer init
                
        self.relu = nn.ReLU()
        
        self.unfold = nn.Unfold(kernel_size=3, stride=1, padding=1)
        
    
        for modules in [self.mlp_2]:
            for l in modules.modules():
                if isinstance(l, nn.Linear):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
        
    def forward(self, x):
        activated_correlation = torch.where(F.normalize(x) > 0, x, torch.zeros_like(x))
        score_tvl = activated_correlation.mean(dim=[2, 3]).unsqueeze(0)
        x = self.select(x) 
        x = x.mean(dim=[2, 3]).unsqueeze(0)
        x = torch.cat((x, score_tvl))
        x = self.mlp_2(x.flatten())
        return x[None, ...]
        
        
        