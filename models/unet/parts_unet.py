import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import complex
import numpy as np


class TripleConv(nn.Module):
    def __init__(self, kshape, in_channels, out_channels, org_in_channels=None):
        super().__init__()

        if org_in_channels is None:
            org_in_channels = in_channels

        self.tripleConvBlock = nn.Sequential(
            # same padding
            nn.Conv2d(org_in_channels, out_channels, kernel_size=kshape, padding=1),
            # no BatchNorm in paper mentioned
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kshape, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kshape, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat([x, y], dim=1)

        return self.tripleConvBlock(x)


class TripleConvComplex(nn.Module):
    def __init__(self, kshape, in_channels, out_channels, org_in_channels=None):
        super().__init__()

        if org_in_channels is None:
            org_in_channels = in_channels

        self.tripleConvBlock = nn.Sequential(
            # same padding
            complex.ComplexConv2D(org_in_channels, out_channels, kernel_size=kshape, stride=1, padding=1),
            # complex.Zrelu(),
            nn.ReLU(inplace=True),
            complex.ComplexConv2D(out_channels, out_channels, kernel_size=kshape, stride=1, padding=1),
            # complex.Zrelu(),
            nn.ReLU(inplace=True),
            complex.ComplexConv2D(out_channels, out_channels, kernel_size=kshape, stride=1, padding=1),
            # complex.Zrelu()
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat([x, y], dim=1)

        return self.tripleConvBlock(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # paper uses nearest interpolation
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = TripleConv(kshape=3, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2, x3=None):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if x3 is None:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.cat([x1, x2, x3], dim=1)

        return self.conv(x)


class UpComplex(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # paper uses nearest interpolation
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = TripleConvComplex(kshape=3, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1, x2, x3=None):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if x3 is None:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.cat([x1, x2, x3], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConvComplex(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvComplex, self).__init__()
        self.conv = complex.ComplexConv2D(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class IFFT(nn.Module):
    def __init__(self, dim=2):
        super(IFFT, self).__init__()
        self.dim = dim

    def forward(self, x):
        use_cuda = torch.cuda.is_available()
        # x.shape = (b_s, 24, 218, 170)
        x = torch.stack((x[:, ::2, :, :], x[:, 1::2, :, :]), dim=-1)
        # x.shape = (b_s, 12, 218, 170, 2)
        x = torch.ifft(x, self.dim)
        # x.shape = (b_s, 12, 218, 170, 2)

        if use_cuda:
            y = (torch.zeros((x.size(0), x.size(1) * 2, x.size(2), x.size(3)))).cuda()
        else:
            y = (torch.zeros((x.size(0), x.size(1) * 2, x.size(2), x.size(3))))

        # re-interleave real and img channels
        y[:, ::2, :, :] = x[:, :, :, :, 0]
        y[:, 1::2, :, :] = x[:, :, :, :, 1]
        return y


class FFT(nn.Module):
    def __init__(self, dim=2):
        super(FFT, self).__init__()
        self.dim = dim

    def forward(self, x):
        use_cuda = torch.cuda.is_available()
        # print("FFT x shape: " , x.shape)
        # x.shape = (b_s, 24, 218, 170)
        x = torch.stack((x[:, ::2, :, :], x[:, 1::2, :, :]), dim=-1)
        # print("FFt reached1")
        # x.shape = (b_s, 12, 218, 170, 2)
        x = torch.fft(x, self.dim)
        # print("FFt reached2")
        # x.shape = (b_s, 12, 218, 170, 2)
        if use_cuda:
            y = (torch.zeros((x.size(0), x.size(1) * 2, x.size(2), x.size(3)))).cuda()
        else:
            y = (torch.zeros((x.size(0), x.size(1) * 2, x.size(2), x.size(3))))
        # print("FFt reached3 y shape ", y.shape)
        # print("FFt reached3 x shape ", x.shape)
        # re-interleave real and img channels
        y[:, ::2, :, :] = x[:, :, :, :, 0]
        y[:, 1::2, :, :] = x[:, :, :, :, 1]
        # print("FFt reached4")

        if use_cuda:
            return y.cuda()
        else:
            return y


class DC_Block(nn.Module):

    def __init__(self, mask_flag):
        super(DC_Block, self).__init__()
        self.apply_mask = mask_flag
        self.fft = FFT(dim=2)

    def forward(self, x, y, kspace_flag):
        if not kspace_flag:
            x = self.fft(x)
        # set all pixels with the same coordinates of the originally sampled kspace-pixels to zero
        # print("Shape x: ", x.shape, "y: ", y.shape)
        if self.apply_mask:  # self.apply_mask:
            # mask = ~(np.abs(y.detach().cpu().numpy()).sum(axis=(0, 1)) == 0)
            mask = (torch.abs(y.sum(axis=1)) == 0)
            # print(mask.shape)
            # mask = torch.ones_like(mask)-mask
            x = x.permute(1, 0, 2, 3)
            x[:, ~mask] = 0
            x = x.permute(1, 0, 2, 3)
            #x = torch.mul(x, mask, )
        # test = ((x + y) == 0)
        # print("Test: ", test.sum())

        x = torch.add(x, y)

        return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
