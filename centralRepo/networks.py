#!/usr/bin/env python
# coding: utf-8
"""
All network architectures: FBCNet, EEGNet, DeepConvNet
@author: Ravikiran Mane
"""

import torch
import torch.nn as nn
from torchsummary import summary
import math
import sys

current_module = sys.modules[__name__]

debug = False

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)


    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

# %% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''

    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''

    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

#%% support of mixConv2d
import torch.nn.functional as F

from typing import Tuple, Optional

def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def _get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

def _same_pad_arg(input_size, kernel_size, stride, dilation):
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dSameExport(nn.Conv2d):
    """ ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    NOTE: This does not currently work with torch.jit.script
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSameExport, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.pad = None
        self.pad_input_size = (0, 0)

    def forward(self, x):
        input_size = x.size()[-2:]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-2:], self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size

        if self.pad is not None:
            x = self.pad(x)
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
            else:
                # dynamic padding
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic

def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        if isinstance(kernel_size, tuple):
            padding = (0,padding)
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        # import  numpy as  np
        # equal_ch = True
        # groups = len(kernel_size)
        # if equal_ch:  # 均等划分通道
        #     in_splits = _split_channels(in_channels, num_groups)
        #     out_splits = _split_channels(out_channels, num_groups)
        # else:  # 指数划分通道
        #     in_splits = _split_channels(in_channels, num_groups)


        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


class ReshapeAndVarLayer(nn.Module):

    def __init__(self, dim, feature,strideFactor=4):
        super(ReshapeAndVarLayer, self).__init__()
        self.dim = dim
        self.feature = feature
        self.strideFactor = strideFactor

    def forward(self, x):
        x = x.reshape(-1, self.feature, self.strideFactor, int(x.shape[3] / self.strideFactor))
        return x.var(dim=self.dim, keepdim=True)


class ReshapeAndLogVarLayer(nn.Module):

    def __init__(self, dim, feature,strideFactor=4):
        super(ReshapeAndLogVarLayer, self).__init__()
        self.dim = dim
        self.feature = feature
        self.strideFactor = strideFactor

    def forward(self, x):
        x = x.reshape(-1, self.feature, self.strideFactor, int(x.shape[3] / self.strideFactor))
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


class resBlock(nn.Module):

    def __init__(self, inChan=9, outChan=9, kernel_size=[(1, 15), (1, 31), (1,63)]):
        super(resBlock, self).__init__()

        self.mixConv2d = nn.Sequential(
            MixedConv2d(in_channels=inChan, out_channels=outChan, kernel_size=kernel_size,
                        stride=1, padding='', dilation=1, depthwise=False, ),
            nn.BatchNorm2d(9),
        )

    def forward(self, x):
        res = x
        x = self.mixConv2d(x)
        x = res + x
        return x


class TSFCNet4a(nn.Module):

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))

    def get_size(self, nChan, nTime):
        x = torch.ones((1, 9, nChan, nTime))
        x_chan = self.chanConv(x)
        x_fb = self.fbConv(x.permute(0, 2, 1, 3))
        x_2d = self.conv2d(x.permute(0, 2, 1, 3))
        out = torch.cat((x_chan, x_fb, x_2d), dim=1)
        return out.size()

    def __init__(self, nChan, nTime, nClass=2, nBands=9, m=36, dropoutP=0.5,
                 temporalLayer='LogVarLayer', strideFactor=4, doWeightNorm=True, *args, **kwargs):
        super(TSFCNet4a, self).__init__()

        self.nBands = nBands
        self.m = m
        self.nClass = nClass
        self.nChan = nChan
        self.outFeature = m

        self.res = resBlock(9,9, [(1, 3), (1, 5)])

        self.chanConv = nn.Sequential(
            nn.Conv2d(9, self.outFeature, kernel_size=(self.nChan, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(self.outFeature),
            nn.ELU(),
            ReshapeAndVarLayer(dim=3, feature=self.outFeature, strideFactor=10),
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout(p=dropoutP),
            nn.Flatten(start_dim=1),
        )

        self.fbConv = nn.Sequential(
            nn.Conv2d(self.nChan, self.outFeature, kernel_size=(9, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(self.outFeature),
            nn.ELU(),
            ReshapeAndVarLayer(dim=3, feature=self.outFeature, strideFactor=10),
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout(p=dropoutP),
            nn.Flatten(start_dim=1),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(self.nChan, self.outFeature, kernel_size=(2, 16), stride=(1, 2), padding=0),
            nn.BatchNorm2d(self.outFeature),
            nn.ELU(),
            VarLayer(dim=3),
            nn.AvgPool2d(kernel_size=(8, 1), stride=(1, 1)),
            nn.Dropout(p=dropoutP),
            nn.Flatten(start_dim=1),
        )
        size = self.get_size(nChan, nTime)
        self.fc = self.LastBlock(size[1], self.nClass, doWeightNorm=True)

    def forward(self, x):
        x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x = self.res(x)
        x_chan = self.chanConv(x)
        x_fb = self.fbConv(x.permute(0, 2, 1, 3))
        x_2d = self.conv2d(x.permute(0, 2, 1, 3))
        f = torch.cat((x_chan, x_2d, x_fb), dim=1)
        out = self.fc(f)
        return out, f

class TSFCNet4b(nn.Module):

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))

    def get_size(self, nChan, nTime):
        x = torch.ones((1, 9, nChan, nTime))
        x_chan = self.chanConv(x)
        x_fb = self.fbConv(x.permute(0, 2, 1, 3))
        x_2d = self.conv2d(x.permute(0, 2, 1, 3))
        out = torch.cat((x_chan, x_fb, x_2d), dim=1)
        return out.size()

    def __init__(self, nChan, nTime, nClass=2, nBands=9, m=36, dropoutP=0.5,
                 temporalLayer='LogVarLayer', strideFactor=4, doWeightNorm=True, *args, **kwargs):
        super(TSFCNet4b, self).__init__()

        self.nBands = nBands
        self.m = m
        self.nClass = nClass
        self.nChan = nChan
        self.outFeature = m

        self.res = resBlock(9,9, [(1, 15), (1, 31), (1, 63),(1, 125)])

        self.chanConv = nn.Sequential(
            nn.Conv2d(9, self.outFeature, kernel_size=(self.nChan, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(self.outFeature),
            nn.ELU(),
            ReshapeAndVarLayer(dim=3, feature=self.outFeature, strideFactor=10),
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout(p=dropoutP),
            nn.Flatten(start_dim=1),
        )

        self.fbConv = nn.Sequential(
            nn.Conv2d(self.nChan, self.outFeature, kernel_size=(9, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(self.outFeature),
            nn.ELU(),
            ReshapeAndVarLayer(dim=3, feature=self.outFeature, strideFactor=10),
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout(p=dropoutP),
            nn.Flatten(start_dim=1),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(self.nChan, self.outFeature, kernel_size=(2, 16), stride=(1, 2), padding=0),
            nn.BatchNorm2d(self.outFeature),
            nn.ELU(),
            VarLayer(dim=3),
            nn.AvgPool2d(kernel_size=(8, 1), stride=(1, 1)),
            nn.Dropout(p=dropoutP),
            nn.Flatten(start_dim=1),
        )
        size = self.get_size(nChan, nTime)
        self.fc = self.LastBlock(size[1], self.nClass, doWeightNorm=True)

    def forward(self, x):
        x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x = self.res(x)
        x_chan = self.chanConv(x)
        x_fb = self.fbConv(x.permute(0, 2, 1, 3))
        x_2d = self.conv2d(x.permute(0, 2, 1, 3))
        f = torch.cat((x_chan, x_2d, x_fb), dim=1)
        out = self.fc(f)
        return out, f