import importlib
import torch.nn.functional as F
import torch.nn as nn
#from parts_unet import *
from . import parts_unet
importlib.reload(parts_unet)
from utils import complex


class WNet(nn.Module):

    def __init__(self, in_channels, kspace_flag=False, architecture='ii', mask_flag=True, complex_flag=False):
        super(WNet, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture

        self.unet_blocks = []
        self.ifft_block = parts_unet.IFFT(dim=2)
        self.dc_block = parts_unet.DC_Block(mask_flag)

        for i, c in enumerate(architecture):
            self.unet_blocks.append(UNetFront(in_channels, in_channels, complex_flag=complex_flag))
            print('unet_complex=', str(complex_flag), '_block added')

        self.unet_blocks = nn.ModuleList(self.unet_blocks)

    def forward(self, x):
        kspace_flag = {'i': False, 'k': True}
        y = x.clone()  # initially, x is in kspace

        for i, c in enumerate(self.architecture):
            if c == 'i':
                x = self.ifft_block(x)
                x, _ = self.unet_blocks[i](x)

            x = self.dc_block(x, y, kspace_flag[c])

        if self.architecture[-1] == 'i':
            x = self.ifft_block(x)

        return x


class WNetDense(nn.Module):

    def __init__(self, in_channels, kspace_flag=False, architecture='ii', mask_flag=True, complex_flag=False):
        super(WNetDense, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture

        self.unet_blocks = []
        self.ifft_block = parts_unet.IFFT(dim=2)
        self.dc_block = parts_unet.DC_Block(mask_flag)
        front_block = {'i': True,
                       'k': True}  # flag to determine if front block or middle block is needed for that domain

        for i, c in enumerate(architecture):
            if front_block[c]:
                self.unet_blocks.append(UNetFront(in_channels, in_channels, complex_flag=complex_flag))
                print('unet_front_block added complex=', str(complex_flag))
                front_block[c] = False
            else:
                self.unet_blocks.append(UNetMiddle(in_channels, in_channels, complex_flag=complex_flag))
                print('unet_middle/end_block added complex=', str(complex_flag))

        self.unet_blocks = nn.ModuleList(self.unet_blocks)

    def forward(self, x):
        kspace_flag = {'i': False, 'k': True}
        z = {'i': None, 'k': None}
        y = x.clone()  # initially, x is in kspace
        for i, c in enumerate(self.architecture):
            if c == 'i':
                x = self.ifft_block(x)
            if z[c] is None:  # if first block of that domain there are no incoming skips
                x, z[c] = self.unet_blocks[i](x)
            else:
                x, z[c] = self.unet_blocks[i](x, z[c])

            x = self.dc_block(x, y, kspace_flag[c])

        if self.architecture[-1] == 'i':
            x = self.ifft_block(x)

        return x


class UNetFront(nn.Module):
    
    def __init__(self, org_in_channels, out_channels, kshape=3, complex_flag=True):
        super(UNetFront, self).__init__()

        self.pool = complex.ComplexMaxPool2D(kernel_size = 2) if complex_flag \
            else nn.MaxPool2d(2)
        # shape=(b_size,channels,218,170)
        self.down1 = parts_unet.TripleConvComplex(kshape, org_in_channels, 48) if complex_flag \
            else parts_unet.TripleConv(kshape, org_in_channels, 48)
        # shape=(b_size,channels,109,85)
        self.down2 = parts_unet.TripleConvComplex(kshape, 48, 64) if complex_flag \
            else parts_unet.TripleConv(kshape, 48, 64)
        # shape=(b_size,channels,54,42)
        self.down3 = parts_unet.TripleConvComplex(kshape, 64, 128)if complex_flag \
            else parts_unet.TripleConv(kshape, 64, 128)
        # shape=(b_size,channels,27,21)
        self.down4 = parts_unet.TripleConvComplex(kshape, 128, 256) if complex_flag \
            else parts_unet.TripleConv(kshape, 128, 256)

        self.up1 = parts_unet.UpComplex(384, 128) if complex_flag \
            else parts_unet.Up(384, 128)

        self.up2 = parts_unet.UpComplex(192, 64) if complex_flag \
            else parts_unet.Up(192, 64)

        self.up3 = parts_unet.UpComplex(112, 48) if complex_flag \
            else parts_unet.Up(112, 48)
        
        self.outc = parts_unet.OutConvComplex(48, out_channels) if complex_flag \
            else parts_unet.OutConv(48, out_channels)

    def forward(self, unet_input):      
        # print("First block")
        # print("Input shape: ", unet_input.shape)
        # unet_input=(b_size,channels,218,170)
        conv1 = self.down1(unet_input)
        pool1 = self.pool(conv1)
        # pool1=(b_size,channels,109,85)
        conv2 = self.down2(pool1)
        pool2 = self.pool(conv2)
        # pool2=(b_size,channels,54,42)
        conv3 = self.down3(pool2)
        pool3 = self.pool(conv3)
        # pool3=(b_size,channels,27,21)
        conv4 = self.down4(pool3)
        # conv4=(b_size,channels,27,21)
        # print("Reache din unet")
        conv5 = self.up1(conv4, conv3)
        # conv5=(b_size,channels,54,42)
        conv6 = self.up2(conv5, conv2)
        # conv6=(b_size,channels,109,86)
        conv7 = self.up3(conv6, conv1)
        # print("reached2 in unet")
        conv8 = self.outc(conv7)
        out = conv8 + unet_input
        # print("Reached 3 in unet")
        # print("Out shape Unet: ", out.shape)
        pre_convs_new = [conv1, conv2, conv3, conv4, conv5, conv6, conv7]
        return out, pre_convs_new


class UNetMiddle(nn.Module):
    
    def __init__(self, org_in_channels, out_channels, kshape=3, complex_flag=True):
        super(UNetMiddle, self).__init__()

        self.pool = complex.ComplexMaxPool2D(kernel_size=2) if complex_flag \
            else nn.MaxPool2d(2)
        # shape=(b_size,channels,218,170)
        self.down1 = parts_unet.TripleConvComplex(kshape, org_in_channels+48, 48) if complex_flag \
            else parts_unet.TripleConv(kshape, org_in_channels+48, 48)
        # shape=(b_size,channels,109,85)
        self.down2 = parts_unet.TripleConvComplex(kshape, 48+64, 64) if complex_flag \
            else parts_unet.TripleConv(kshape, 48+64, 64)
        # shape=(b_size,channels,54,42)
        self.down3 = parts_unet.TripleConvComplex(kshape, 64+128, 128) if complex_flag \
            else parts_unet.TripleConv(kshape, 64+128, 128)
        # shape=(b_size,channels,27,21)
        self.down4 = parts_unet.TripleConvComplex(kshape, 128+256, 256) if complex_flag \
            else parts_unet.TripleConv(kshape, 128+256, 256)

        self.up1 = parts_unet.UpComplex(512, 128) if complex_flag \
            else parts_unet.Up(512, 128)

        self.up2 = parts_unet.UpComplex(256, 64) if complex_flag \
            else parts_unet.Up(256, 64)

        self.up3 = parts_unet.UpComplex(160, 48) if complex_flag \
            else parts_unet.Up(160, 48)

        self.outc = parts_unet.OutConvComplex(48, out_channels) if complex_flag \
            else parts_unet.OutConv(48, out_channels)

    def forward(self, unet_input, pre_convs):      
        # print("First block")
        # print("Input shape: ", unet_input.shape)
        # unet_input=(b_size,channels,218,170)
        conv1 = self.down1(unet_input, pre_convs[6])
        pool1 = self.pool(conv1)
        # pool1=(b_size,channels,109,85)
        conv2 = self.down2(pool1, pre_convs[5])
        pool2 = self.pool(conv2)
        # pool2=(b_size,channels,54,42)
        conv3 = self.down3(pool2, pre_convs[4])
        pool3 = self.pool(conv3)
        # pool3=(b_size,channels,27,21)
        conv4 = self.down4(pool3, pre_convs[3])
        # conv4=(b_size,channels,27,21)
        # print("Reache din unet")
        conv5 = self.up1(conv4, conv3, pre_convs[2])
        # conv5=(b_size,channels,54,42)
        conv6 = self.up2(conv5, conv2, pre_convs[1])
        # conv6=(b_size,channels,109,86)
        conv7 = self.up3(conv6, conv1, pre_convs[0])
        # print("reached2 in unet")
        conv8 = self.outc(conv7)
        out = conv8 + unet_input
        # print("Reached 3 in unet")
        # print("Out shape Unet: ", out.shape)
        pre_convs_new = [conv1, conv2, conv3, conv4, conv5, conv6, conv7]
        return out, pre_convs_new
