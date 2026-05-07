import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .crossMamba import DDAM  # Renamed Dynamic Dual-Attention Mamba
from .DDFEformer import DDFE 

##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

##########################################################################
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
class DDFusionBlock(nn.Module):
    """
    Dual-Domain Mamba Fusion Block.
    Fuses Dual-Domain Fusion Feature Extraction (DDFE) 
    and Dynamic Dual-Attention Mamba (DDAM).
    """
    def __init__(self, dim, depth, mixer_kernel_size, local_size, bias, drop_path):
        super(DDFusionBlock, self).__init__()
        
        self.ddfe_branch = DDFE(depth, dim, mixer_kernel_size, local_size)

        self.ddam_branch = DDAM(dim, depth // 2, drop_path)
        
        self.fusion_conv = nn.Conv2d(int(dim * 2), dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1 = self.ddfe_branch(x)
        x2 = self.ddam_branch(x)
        
        out = torch.cat((x1, x2), dim=1)
        return self.fusion_conv(out)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

##########################################################################
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

##########################################################################
class TransMamba(nn.Module):
    """
    TransMamba: Efficient Hierarchical Network for Medical Imaging.
    Utilizes DDFusionBlocks for comprehensive feature extraction.
    """
    def __init__(self,
        inp_channels=128,
        out_channels=3,
        dim=24,
        dim_2=24,
        num_blocks=[1, 1, 1, 1],
        num_refinement_blocks=2,
        bias=False,
        depth=[2, 4, 8, 10],
        local_size=[4, 4, 4, 4],
        drop_path_rate=0.1,
        **kwargs
    ):
        super(TransMamba, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim_2)
        self.final_embed = OverlapPatchEmbed(48, 2)

        # Encoder Layers
        self.encoder_level1 = nn.Sequential(*[
            DDFusionBlock(dim, depth[0], [1, 3, 5, 7], local_size[0], bias, dpr[sum(depth[:0]):sum(depth[:1])]) 
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            DDFusionBlock(int(dim*2**1), depth[1], [1, 3, 5, 7], local_size[1], bias, dpr[sum(depth[:1]):sum(depth[:2])]) 
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim*2**1))

        self.encoder_level3 = nn.Sequential(*[
            DDFusionBlock(int(dim*2**2), depth[2], [1, 3, 5, 7], local_size[2], bias, dpr[sum(depth[:2]):sum(depth[:3])]) 
            for _ in range(num_blocks[2])
        ])

        # Decoder Layers
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            DDFusionBlock(int(dim*2**1), depth[1], [1, 3, 5, 7], local_size[1], bias, dpr[::-1][depth[2]:depth[2]+depth[1]]) 
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[
            DDFusionBlock(int(dim*2**1), depth[0], [1, 3, 5, 7], local_size[0], bias, dpr[::-1][depth[2]+depth[1]:sum(depth[:3])]) 
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            DDFusionBlock(int(dim*2**1), depth[0], [1, 3, 5, 7], local_size[0], bias, dpr[::-1][0:depth[2]]) 
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # Initial Embedding
        inp_enc_level1 = self.patch_embed(inp_img)

        # Encoder Pathway
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # Decoder Pathway with Skip Connections
        inp_dec_level2 = self.up3_2(out_enc_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # Final Processing
        out_dec_level1 = self.final_embed(out_dec_level1)

        return out_dec_level1