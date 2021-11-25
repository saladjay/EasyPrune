import os
from copy import copy
from pathlib import Path

import math
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn


def get_depth(n, gd):
    n = max(round(n * gd), 1) if n > 1 else n  # depth gain
    return n


def get_witdh(x, divisor=8):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


# kernel, padding
def autopad(k, p=None):
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups, log_name
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # x1 = self.conv(x)
        # x2 = self.bn(x1)
        # return self.act(x2)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # x1 = self.cv1(x)
        # x2 = self.cv2(x1)
        # if self.add:
        #     return x + x2
        # else:
        #     return x2
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        # x1 = self.cv1(x)
        # x2 = self.m(x1)
        # x3 = self.cv2(x)
        # x4 = torch.cat([x2,x3],dim=1)
        # return self.cv3(x4)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Focus(nn.Module):
    # Focus wh information into c-space
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    # x(b,c,w,h) -> y(b,4c,w/2,h/2)
    def forward(self, x):
        # print(int(id(x)))
        # x1 = x[...,::2,::2]
        # print(int(id(x1)))
        # x2 = x[...,1::2,::2]
        # print(int(id(x2)))
        # x3 = x[...,::2,1::2]
        # print(int(id(x3)))
        # x4 = x[...,1::2,1::2]
        # print(int(id(x4)))
        # x5 = torch.cat([x1,x2,x3,x4],dim=1)
        # print(int(id(x5)))
        # return self.conv(x5)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[...,::2, 1::2], x[..., 1::2, 1::2]], 1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))




class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)