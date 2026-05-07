import torch
import torch.nn as nn

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class Freq_Fusion(nn.Module):
    def __init__(self, dim, kernel_size=[1,3,5,7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.conv_init_1 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.conv_init_2 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.conv_init_3 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        
        self.FFC = FourierUnit(self.dim*3, self.dim*3)
        self.bn = torch.nn.BatchNorm2d(dim*3)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2, x_3 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x_3 = self.conv_init_3(x_3)
        
        x0 = torch.cat([x_1, x_2, x_3], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))

        return x


class Fused_Fourier_Conv_Mixer(nn.Module):
    def __init__(self, dim, token_mixer_for_gloal=Freq_Fusion, mixer_kernel_size=[1,3,5,7], local_size=8):
        super(Fused_Fourier_Conv_Mixer, self).__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size,
                                                 se_ratio=8, local_size=local_size)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(3*dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 3, 1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_3 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=1 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        
        x_local_1 = self.dw_conv_1(x[0])
        x_local_2 = self.dw_conv_2(x[1])
        x_local_3 = self.dw_conv_3(x[2])
        
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2, x_local_3], dim=1))
        
        x = self.ca_conv(x_gloal)
        x = self.ca(x) * x

        return x


class Prior_Gated_Feed_forward_Network(nn.Module):
    def __init__(self, dim, kernel_size=[1,3,5,7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        super(Prior_Gated_Feed_forward_Network, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.conv_dw = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=3 // 2, groups=dim*2,
                      padding_mode='reflect'),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = self.conv_dw(x)
        x = self.conv_fina(x)
        return x


class DDFEBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, token_mixer=Fused_Fourier_Conv_Mixer, 
                 kernel_size=[1,3,5,7], local_size=8, h=224, w=224, shorcut=True):
        super(DDFEBlock, self).__init__()
        self.dim = dim
        self.norm1 = torch.nn.BatchNorm2d(dim)
        self.norm2 = torch.nn.BatchNorm2d(dim)
        
        self.mixer = token_mixer(dim=self.dim, mixer_kernel_size=kernel_size, local_size=local_size)
        self.ffn = Prior_Gated_Feed_forward_Network(dim=self.dim)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x) 
        x = x + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x) 
        x = x + copy

        return x


class DDFE(nn.Module):  
    def __init__(self, depth=1, dim=64, mixer_kernel_size=[1,3,5,7], local_size=8):
        super(DDFE, self).__init__()
        
        # Init blocks
        self.blocks = nn.Sequential(*[
            DDFEBlock(
                dim=dim,
                norm_layer=nn.BatchNorm2d,
                token_mixer=Fused_Fourier_Conv_Mixer,
                kernel_size=mixer_kernel_size,
                local_size=local_size
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
