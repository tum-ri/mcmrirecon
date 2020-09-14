import torch
import torch.nn as nn
import numpy as np

from math import pi
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d
from torch.nn.init import _calculate_fan_in_and_fan_out
import sys
sys.path.append("..")
from configs.config_wnet import config

'''
Pytorch implementation of: https://github.com/MRSRL/complex-networks-release/blob/master/complex_utils.py
Implementation related to the paper "Complex-Valued Convolutional Neural Networks for MRI Reconstruction" 
by Elizabeth K. Cole et. al: https://arxiv.org/abs/2004.01738
'''


class ComplexConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2D, self).__init__()

        in_channels = in_channels // 2
        out_channels = out_channels // 2

        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        if config['complex_weight_init']:
            self.conv_r.weight = torch.nn.Parameter(complex_init_individual(self.conv_r))
            self.conv_i.weight = torch.nn.Parameter(complex_init_individual(self.conv_i))

    def forward(self, x_complex):  # (batch_size, channels, image_height, image_width)
        real_out = self.conv_r(x_complex[:, ::2]) - self.conv_i(x_complex[:, 1::2])
        imag_out = self.conv_r(x_complex[:, 1::2]) + self.conv_i(x_complex[:, ::2])

        b, c, h, w = real_out.shape
        if torch.cuda.is_available():
            output = torch.empty(b, c*2, h, w).cuda()
        else:
            output = torch.empty(b, c * 2, h, w)
        output[:, ::2] = real_out
        output[:, 1::2] = imag_out
        return output


class ComplexConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConvTranspose2D, self).__init__()

        in_channels = in_channels // 2
        out_channels = out_channels // 2

        self.conv_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x_complex):   # (batch_size, channels, image_height, image_width)
        real_out = self.conv_r(x_complex[:, ::2]) - self.conv_i(x_complex[:, 1::2])
        imag_out = self.conv_r(x_complex[:, 1::2]) + self.conv_i(x_complex[:, ::2])

        b, c, h, w = real_out.shape
        if torch.cuda.is_available():
            output = torch.empty(b, c*2, h, w).cuda()
        else:
            output = torch.empty(b, c * 2, h, w)
        output[:, ::2] = real_out
        output[:, 1::2] = imag_out
        return output


class ComplexMaxPool2D(nn.Module):
    def __init__(self, kernel_size, padding=0, dilation=1, return_indices=True, ceil_mode=False):
        super(ComplexMaxPool2D, self).__init__()

        self.max_pool = MaxPool2d(kernel_size=kernel_size,
                                  padding=padding,
                                  dilation=dilation,
                                  return_indices=return_indices,
                                  ceil_mode=ceil_mode)

    def forward(self, x_complex):
        abs_pool = torch.sqrt(torch.square(x_complex[:,::2]) + torch.square(x_complex[:,1::2]))
        #abs_pool = x_complex.cpu().detach().numpy()
        #abs_pool = np.abs(abs_pool[:, ::2]+1j*abs_pool[:, 1::2])
        #abs_pool = torch.from_numpy(abs_pool.repeat(2, axis=1)).cuda()
        abs_pool, indices = self.max_pool(abs_pool)
        indices = indices.repeat_interleave(2, dim=1)
        
        flattened_tensor = x_complex.flatten(start_dim=2)
        output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        return output



class Zrelu(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_complex):
        phase = torch.atan2(input_complex[:, 1::2], input_complex[:, ::2])
        # if phase > pi/2, throw it away and set comp equal to 0
        gt = torch.gt(phase, pi / 2)
        input_complex[:, ::2][gt] = 0
        input_complex[:, 1::2][gt] = 0
        # if phase < 0, throw it away and set output equal to 0
        st = ~torch.ge(phase, 0)
        input_complex[:, ::2][st] = 0
        input_complex[:, 1::2][st] = 0

        return input_complex


def Cardioid(input_complex):
    phase = torch.atan2(input_complex[:, 1::2], input_complex[:, ::2])
    scale = 0.5 * (1 + torch.cos(phase))
    real_out = input_complex[:, ::2] * scale
    imag_out = input_complex[:, 1::2] * scale
    output = torch.empty_like(input_complex)
    output[:, ::2] = real_out
    output[:, 1::2] = imag_out

    return output


def weights_init(m, criterion='glorot'):
    classname = m.__class__.__name__

    if classname.find('Conv2D') != -1:
        complex_init(m.weight, criterion)


def complex_init(m, criterion):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
    if criterion == 'glorot':
        s = 1. / (fan_in + fan_out)
    elif criterion == 'he':
        s = 1. / fan_in
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = np.random.RandomState(1337)
    modulus = rng.rayleigh(scale=s, size=m.weight.shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=m.weight.shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)
    weight = np.concatenate([weight_real, weight_imag], axis=-1)
    return torch.from_numpy(weight)


def complex_init_individual(m, criterion='glorot'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
    if criterion == 'glorot':
        s = 1. / (fan_in + fan_out)
    elif criterion == 'he':
        s = 1. / fan_in
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    b, c, h, w = m.weight.shape
    rng = np.random.RandomState(1337)
    modulus = rng.rayleigh(scale=s, size=(b, c // 2, h, w))
    phase = rng.uniform(low=-np.pi, high=np.pi, size=(b, c // 2, h, w))
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)
    weight = np.empty(m.weight.shape)
    weight[:, ::2, :, :] = weight_real
    weight[:, 1::2, :, :] = weight_imag
    weight = torch.from_numpy(weight).float()
    return weight
