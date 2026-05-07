# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    """ Multilayer Perceptron (MLP) used in Transformer blocks. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    Supports 2-modal feature fusion.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv_c = nn.Linear(dim * 2, dim * 6, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_x = nn.Linear(dim * 2, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x, y: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # Concatenate features of x and y
        x_y = torch.cat([x, y], dim=2)
        B_c, N_c, C_c = x_y.shape

        # Generate Q, K, V from concatenated features
        qkv = self.qkv_c(x_y).reshape(B_c, N_c, 3, self.num_heads, C_c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_c // nW, nW, self.num_heads, N_c, N_c) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_c, N_c)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x_out = (attn @ v).transpose(1, 2).reshape(B_c, N_c, C_c)
        
        # Compress the concatenated feature dimension back to the original dimension
        x_out = self.proj_x(x_out)
        x_out = self.proj_drop(x_out)

        return x_out

class WindowAttention_Three(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    Supports 3-modal feature fusion (past, current, next).
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv_c = nn.Linear(dim * 3, dim * 9, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_x_three = nn.Linear(dim * 3, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, z, mask=None):
        x_y = torch.cat([x, y, z], dim=2)
        B_c, N_c, C_c = x_y.shape

        qkv = self.qkv_c(x_y).reshape(B_c, N_c, 3, self.num_heads, C_c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_c // nW, nW, self.num_heads, N_c, N_c) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_c, N_c)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x_out = (attn @ v).transpose(1, 2).reshape(B_c, N_c, C_c)
        x_out = self.proj_x_three(x_out)
        x_out = self.proj_drop(x_out)

        return x_out

class SwinTransformerBlock(nn.Module):
    r""" Multi-Modal Swin Transformer Block. """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn_three = WindowAttention_Three(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # Generate attention mask for shifted window to prevent cross-window attention
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, y, z, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut_x, shortcut_y, shortcut_z = x, y, z

        # Reshape and normalize
        x = self.norm1(x).view(B, H, W, C)
        y = self.norm1(y).view(B, H, W, C)
        z = self.norm1(z).view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_z = torch.roll(z, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x, shifted_y, shifted_z = x, y, z

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        y_windows = window_partition(shifted_y, self.window_size).view(-1, self.window_size * self.window_size, C)
        z_windows = window_partition(shifted_z, self.window_size).view(-1, self.window_size * self.window_size, C)

        # Calculate attention
        mask = self.attn_mask if self.input_resolution == x_size else self.calculate_mask(x_size).to(x.device)

        # Past (x) referring to Current (y)
        attn_windows = self.attn(x_windows, y_windows, mask=mask)
        # Next (z) referring to Current (y)
        attn_windows_z = self.attn(z_windows, y_windows, mask=mask)
        # Current (y) referring to Past (x) and Next (z)
        attn_windows_y = self.attn_three(y_windows, x_windows, z_windows, mask=mask)

        # Merge windows and reverse cyclic shift
        def reverse_process(attn_w):
            attn_w = attn_w.view(-1, self.window_size, self.window_size, C)
            shifted = window_reverse(attn_w, self.window_size, H, W)
            if self.shift_size > 0:
                out = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                out = shifted
            return out.view(B, H * W, C)

        x = reverse_process(attn_windows)
        y = reverse_process(attn_windows_y)
        z = reverse_process(attn_windows_z)

        # FFN (LN + MLP)
        x = shortcut_x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        y = shortcut_y + self.drop_path(y)
        y = y + self.drop_path(self.mlp(self.norm2(y)))

        z = shortcut_z + self.drop_path(z)
        z = z + self.drop_path(self.mlp(self.norm2(z)))

        return x, y, z

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage. """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, y, z, x_size):
        for blk in self.blocks:
            x, y, z = blk(x, y, z, x_size)
        return x, y, z

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB). """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim, input_resolution=input_resolution,
                                         depth=depth, num_heads=num_heads,
                                         window_size=window_size, mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                         norm_layer=norm_layer, downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, y, z, x_size):
        x_rstb, y_rstb, z_rstb = self.residual_group(x, y, z, x_size)
        
        # Un-embed -> Conv -> Embed + Residual
        x_rstb = self.patch_embed(self.conv(self.patch_unembed(x_rstb, x_size))) + x
        y_rstb = self.patch_embed(self.conv(self.patch_unembed(y_rstb, x_size))) + y
        z_rstb = self.patch_embed(self.conv(self.patch_unembed(z_rstb, x_size))) + z

        return x_rstb, y_rstb, z_rstb

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

class Upsample(nn.Sequential):
    """Upsample module."""
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    """ Lightweight Upsample module (1conv + 1pixelshuffle). """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        m = [nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1), nn.PixelShuffle(scale)]
        super(UpsampleOneStep, self).__init__(*m)

# ---------------------------------------------------------------------------------------------------------

class Swin_multi_modal_block(nn.Module):
    """ Main Multi-Modal Swin Block Network (Process Past, Current, Next Features). """
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=128, depths=[4], num_heads=[4], window_size=8, 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv', **kwargs):
        super(Swin_multi_modal_block, self).__init__()

        num_in_ch = num_out_ch = in_chans
        num_feat = 512
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.ape = ape

        rgb_mean = (0.4488, 0.4371, 0.4040) if in_chans == 3 else (0,)
        self.mean = torch.Tensor(rgb_mean).view(1, in_chans, 1, 1) if in_chans == 3 else torch.zeros(1, 1, 1, 1)
        self.mean_y = self.mean.clone()
        self.mean_z = self.mean.clone()

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build Residual Swin Transformer blocks (RSTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=depths[i_layer], num_heads=num_heads[i_layer],
                         window_size=window_size, mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer, downsample=None,
                         use_checkpoint=use_checkpoint, img_size=img_size,
                         patch_size=patch_size, resi_connection=resi_connection)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # Upsampling modules
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x, y, z):
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)
        y = self.patch_embed(y)
        z = self.patch_embed(z)

        if self.ape:
            x, y, z = x + self.absolute_pos_embed, y + self.absolute_pos_embed, z + self.absolute_pos_embed

        x, y, z = self.pos_drop(x), self.pos_drop(y), self.pos_drop(z)

        for layer in self.layers:
            x, y, z = layer(x, y, z, x_size)

        x, y, z = self.norm(x), self.norm(y), self.norm(z)

        x = self.patch_unembed(x, x_size)
        y = self.patch_unembed(y, x_size)
        z = self.patch_unembed(z, x_size)

        return x, y, z

    def forward(self, x, y, z):
        H, W = x.shape[2:]

        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        y = self.check_image_size(y)
        self.mean_y = self.mean_y.type_as(y)
        y = (y - self.mean_y) * self.img_range

        z = self.check_image_size(z)
        self.mean_z = self.mean_z.type_as(z)
        z = (z - self.mean_z) * self.img_range

        x, y, z = self.forward_features(x, y, z)

        # Restore original data ranges
        x = x / self.img_range + self.mean
        y = y / self.img_range + self.mean_y
        z = z / self.img_range + self.mean_z

        return x[:, :, :H*self.upscale, :W*self.upscale], y[:, :, :H*self.upscale, :W*self.upscale], z[:, :, :H*self.upscale, :W*self.upscale]