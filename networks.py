import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Custom modules
from .residual_swin_transformers_multi_modal import Swin_multi_modal_block
from .transMamba_arch import TransMamba

class TemporalConv(nn.Module):
    """ Temporal feature extraction module using 1x1 convolutions. """
    def __init__(self, in_nc, out_nc):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_nc),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc + out_nc, out_channels=out_nc, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_nc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat((x, x1), dim=1)
        x3 = self.conv2(x2)
        x4 = torch.cat((x2, x3), dim=1)
        return x4


class SpatialConv(nn.Module):
    """ Spatial feature extraction module using 3x3 convolutions. """
    def __init__(self, in_nc, out_nc):
        super(SpatialConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_nc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv1(x)


class MIAStage(nn.Module):
    """ Multi-stage Information Aggregation (MIA) via alternating temporal and spatial convolutions. """
    def __init__(self):
        super(MIAStage, self).__init__()
        self.temporal1 = TemporalConv(200, 179)
        self.spatial1 = SpatialConv(558, 256)
        
        self.temporal2 = TemporalConv(256, 147)
        self.spatial2 = SpatialConv(550, 224)
        
        self.temporal3 = TemporalConv(224, 115)
        self.spatial3 = SpatialConv(454, 192)
        
        self.temporal4 = TemporalConv(192, 83)
        self.spatial4 = SpatialConv(358, 160)
        
        self.temporal5 = TemporalConv(160, 51)
        self.spatial5 = SpatialConv(262, 128)

    def forward(self, x):
        x1 = self.temporal1(x)
        x2 = self.spatial1(x1)
        
        x3 = self.temporal2(x2)
        x4 = self.spatial2(x3)
        
        x5 = self.temporal3(x4)
        x6 = self.spatial3(x5)
        
        x7 = self.temporal4(x6)
        x8 = self.spatial4(x7)
        
        x9 = self.temporal5(x8)
        x10 = self.spatial5(x9)
        return x10


class SpatioTemporalFusionNet(nn.Module):
    """ 
    Spatio-Temporal Fusion Network.
    Fuses past, current, and next features using MIA stages, Swin Transformer, and TransMamba.
    """
    def __init__(self, n_channels=200, G0=64, kSize=1, D=3, T=128, C=4, G=64, dilateSet=[1, 1, 1, 1]):
        super(SpatioTemporalFusionNet, self).__init__()
        
        self.D = D
        self.C = C
        self.kSize = kSize
        self.dilateSet = dilateSet
        self.T = T

        # Multi-modal feature fusion blocks
        self.swin_block = Swin_multi_modal_block(img_size=256, patch_size=1, in_chans=T, embed_dim=T)
        self.transmamba = TransMamba()
        
        # MIA Feature extractors for distinct temporal states
        self.mia1 = MIAStage()
        self.mia2 = MIAStage()
        self.mia3 = MIAStage()
        

    def forward(self, x_feature_past, x_feature_current, x_feature_next):
        # Extract multi-stage features
        past = self.mia1(x_feature_past)
        current = self.mia2(x_feature_current)
        next_feat = self.mia3(x_feature_next)

        # Swin Transformer based fusion
        _, transformer_out_tag, _ = self.swin_block(past, current, next_feat)

        # Temporal sequence modeling via TransMamba
        out = self.transmamba(transformer_out_tag)

        return out


if __name__ == '__main__':
    pass