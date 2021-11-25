import os

import torch.nn.functional as F
import torch
from enum import Enum

from EasyPrune.forwardlog import forwardlog
import traceback

log = forwardlog()
DEBUG = False
INLINE = False

REGISTERED_LIST = [
    ###################TensorRT Support############################
    'Conv2d', 'ConvTranspose2d',
    'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d', 'ZeroPad2d',
    'Linear',
    'ReLU', 'LeakyReLU', 'Sigmoid', "Softmax", "SiLU",
    'MaxPool2d', 'AvgPool2d',
    'BatchNorm2d', 'BatchNorm1d', 'BatchNorm3d', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
    #################TensorRT not support class###############
    'AdaptiveAvgPool2d'
]

raw_operation_dict = {}
raw__add__ = None
raw__sub__ = None
raw__mul__ = None
raw__permute__ = None
raw__expand_as__ = None
raw_get_item = None


# nn.Conv2d-->F.conv2d
def _conv2d(raw, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    global INLINE
    x = raw(input, weight, bias, stride, padding, dilation, groups)
    INLINE = True
    name = log.add_blob("conv2d_", input, x)
    log.add_weight(f'{name}.weight', weight=weight)
    if bias is not None:
        log.add_weight(f'{name}.bias', weight=bias)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# nn.ReLU----->F.relu
def _relu(raw, input, inplace=False):
    global INLINE
    x = raw(input, inplace)
    INLINE = True
    name = log.add_blob('relu_', input, x)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# nn.LeakyReLU---->F.leaky_relu
def _leaky_relu(raw, input, negative_slope=0.01, inplace=False):
    global INLINE
    x = raw(input, negative_slope, inplace)
    INLINE = True
    name = log.add_blob("leaky_relu_", input, x)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# nn.Sigmoid----->torch.sigmoid
def _sigmoid(raw, input):
    global INLINE
    x = raw(input)
    INLINE = True
    name = log.add_blob("sigmoid_", input, x)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# nn.SiLU----->F.silu
def _silu(raw, input, inplace=False):
    global INLINE
    x = raw(input, inplace)
    INLINE = True
    name = log.add_blob('silu_', input, x)
    if DEBUG:
        print(name)
    INLINE = False
    return x

# torch.nn.PReLU----->F.prelu
def _prelu(raw, input, weights):
    global INLINE
    x = raw(input, weights)
    name = log.add_blob('prelu_', input, x)
    if DEBUG:
        print(name)
    INLINE = False
    return x

# nn.MaxPool2d----->F.max_pool2d
def _max_pool2d(raw, *args, **kwargs):
    # args = (input, kernel, stride, padding, dilation, ceil_mode, return_indices)
    global INLINE
    x = raw(*args, **kwargs)
    INLINE = True
    name = log.add_blob("max_pool2d_", args[0], x)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# nn.AvgPool2d------>F.avg_pool2d
def _avg_pool2d(raw, input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                divisor_override=None):
    global INLINE
    x = raw(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
    INLINE = True
    name = log.add_blob("avg_pool2d_", input, x)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# nn.Linear---->F.linear
def _linear(raw, input, weight, bias=None):
    global INLINE
    x = raw(input, weight, bias)
    INLINE = True
    name = log.add_blob("linear_", input, x)
    log.add_weight(f'{name}.weight', weight)

    if bias is not None:
        log.add_weight(f'{name}.bias', bias)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# torch.flatten
# one temp solution, maybe error in some scence, here just for test resnet50
# TODO: more Robust for many scence
# other way [torch.reshpe | Tensor.view <<=======>> addShuffle] also OK!
# each way can go to Rome, just follow your favorite !
def _flatten(raw, input, start_dim=0, end_dim=-1):
    global INLINE
    x = raw(input, start_dim, end_dim)
    INLINE = True
    name = log.add_blob("reduce(flatten)_", input, x)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# torch.cat
def _cat(raw, inputs, dim=0):
    global INLINE
    x = raw(inputs, dim)
    INLINE = True
    name = log.add_blob("cat_", inputs, x)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# nn.Softmax--->F.softmax
def _softmax(raw, input, dim=None, _stacklevel=3, dtype=None):
    global INLINE
    x = raw(input, dim, _stacklevel, dtype)
    INLINE = True
    name = log.add_blob("softmax_", input, x)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]----> F.batch_norm----> torch.batch_norm
def _batch_norm(raw, input, weight, bias, running_mean, running_var, training, momentum, eps,
                torch_backends_cudnn_enabled):
    global INLINE
    x = raw(input, weight, bias, running_mean, running_var, training, momentum, eps, torch_backends_cudnn_enabled)
    INLINE = True
    name = log.add_blob("BN_", input, x)
    log.add_weight(f'{name}.weight', weight)
    log.add_weight(f'{name}.bias', bias)
    log.add_weight(f'{name}.running_mean', running_mean)
    log.add_weight(f'{name}.running_var', running_var)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# ['InstanceNorm1d','InstanceNorm2d', 'InstanceNorm3d']------>F.instance_norm---->torch.instance_norm
def _instance_norm(raw, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps,
                   torch_backends_cudnn_enabled):
    global INLINE
    x = raw(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps,
            torch_backends_cudnn_enabled)
    INLINE = True
    name = log.add_blob("IN_", input, x)
    log.add_weight(f'{name}.weight', weight)
    log.add_weight(f'{name}.bias', bias)
    log.add_weight(f'{name}.running_mean', running_mean)
    log.add_weight(f'{name}.running_var', running_var)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# ConvTranspose2d---->F.conv_transpose2d
def _conv_transpose2d(raw, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    global INLINE
    x = raw(input, weight, bias, stride, padding, output_padding, groups, dilation)
    INLINE = True
    name = log.add_layer("Deconv2d_", input, x)
    log.add_weight(f'{name}.weight', weight)
    if bias is not None:
        log.add_weight(f'{name}.bias', bias)
    if DEBUG:
        print(name)
    INLINE = False
    return x


# ['ConstantPad1d','ConstantPad2d', 'ConstantPad3d', 'ZeroPad2d']----->F.pad
# here for pytorch deconv outputpadding param, temp use
# TODO: more flexible or DIY use, maybe you can use config.py? anyway it's up to you like and make sure right!
def _pad(raw, input, pad, mode="constant", value=0):
    global INLINE
    x = raw(input, pad, mode, value)
    INLINE = True
    name = log.add_blob("pad_", input, x)
    if DEBUG:
        print(name)

    return x


# torch.topk
def _topk(raw, input, k, dim=None, largest=True, sorted=True):
    global INLINE
    x = raw(input, k, dim, largest, sorted)
    name = log.add_blob("topk_", input, x)
    if DEBUG:
        print(name)
    return x


# torch.argmax
def _argmax(raw, input, dim, keepdim=False):
    global INLINE
    x = raw(input, dim, keepdim)
    name = log.add_blob("argmax_", input, x)
    if DEBUG:
        print(name)
    return x


# F.interpolate ,  here we use Down/up samples the input to either the given size (resize, upsampling, downsampling)
def _interpolate(raw, input, size=None, scale_factor=None, mode='nearest', align_corners=None,
                 recompute_scale_factor=None):
    global INLINE
    x = raw(input, size, scale_factor, mode, align_corners, recompute_scale_factor)
    name = log.add_blob("interpolate_", input, x)
    if DEBUG:
        print(name)
    return x


# unaryop
def _unaryop(raw, style, input):
    global INLINE
    x = raw(input)
    name = log.add_layer("unaryop_", input, x)
    if DEBUG:
        print(name)
    return x


# _add
def _add(input, *args):
    global INLINE
    x = raw__add__(input, *args)
    name = log.add_blob("add_", [input, args[0]], x)
    if DEBUG:
        print(name)
    return x


# _sub
def _sub(input, *args):
    global INLINE
    x = raw__sub__(input, *args)
    name = log.add_blob("sub_", [input, args[0]], x)
    if DEBUG:
        print(name)
    return x


# _expand_as
def _expand_as(input, *args):
    global INLINE
    x = raw__expand_as__(input, *args)
    name = log.add_blob("expand_as_", input, x)
    if DEBUG:
        print(name)
    return x


# _permute
# TODO merge with reshape layer to shuffle layer
def _permute(input, *args):
    global INLINE
    x = raw__permute__(input, *args)
    name = log.add_blob("permute_", input, x)
    if DEBUG:
        print(name)
    return x


# torch.div
def _div(raw, input, other):
    global INLINE
    x = raw(input, other)
    name = log.add_blob("div_", [input, other], x)
    if DEBUG:
        print(name)
    return x


# torch.split
def _split(raw, tensor, split_size_or_sections, dim=0):
    global INLINE
    x = raw(tensor, split_size_or_sections, dim)
    name = log.add_blob("split_", tensor, x)
    if DEBUG:
        print(name)
    return x


# torch.reshape
def _reshape(raw, input, shape):
    global INLINE
    x = raw(input, shape)
    name = log.add_blob("reshape_", input, x)
    if DEBUG:
        print(name)
    return x


# torch.matmul
def _matmul(raw, input, other):
    global INLINE
    x = raw(input, other)
    name = log.add_blob("matmul_", [input, other], x)
    if DEBUG:
        print(name)
    return x


# nn.AdaptiveAvgPool2d----->F.adaptive_avg_pool2d
# tensorrt not support , just pytorch test
def _adaptive_avg_pool2d(raw, input, output_size):
    global INLINE
    x = raw(input, output_size)
    name = log.add_blob("adaptive_avg_pool2d_", input, x)
    if DEBUG:
        print(name)
    return x


# a[11 dim weights] * b[tensor] for scale layer, just test
# TODO: You can add it manually after generating JSON. and wgt. This is just a format note
# or you can complete it by yourself way , maybe config.py can be useful?
def _mul(input, *args):
    global INLINE
    x = raw__mul__(input, *args)
    name = log.add_blob("mul_", input, x)
    if DEBUG:
        print(name)
    return x


# update at 2021/10/26
#
def _get_item(*args, **kwargs):
    global INLINE
    x = raw_get_item(*args, **kwargs)
    INLINE = True
    name = log.add_blob('slice_', args[0], x)
    if DEBUG:
        print(name)
    INLINE = False
    return x


class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        # replace the raw function to replace funtion
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        global INLINE
        if INLINE:
            return self.raw(*args, **kwargs)
        else:
            if DEBUG:
                print()
                print("*" * 200)
                for stack in traceback.walk_stack(None):
                    if 'self' in stack[0].f_locals:
                        layer = stack[0].f_locals['self']
                        if type(layer).__name__ in REGISTERED_LIST:
                            print(layer)
                            break
            return self.obj(self.raw, *args, **kwargs)


UnaryOperationStyle = Enum(("UnaryOperationStyle"),
                           ("kEXP", "kLOG", "kSQRT", "kRECIP", "kABS", "kNEG", "kSIN", "kCOS", "kTAN", "kSINH", "kCOSH",
                            "kASIN", "kACOS", "kATAN", "kASINH", "kACOSH", "kATANH", "kCEIL", "kFLOOR", "kERF", "kNOT"),
                           start=0)


class UnaryOperation(object):
    def __init__(self, raw, replace, style, **kwargs):
        # replace the raw function to replace funtion
        self.obj = replace
        self.raw = raw
        self.style = style

    def __call__(self, *args, **kwargs):
        global INLINE
        if INLINE:
            return self.raw(*args, **kwargs)
        else:
            if DEBUG:
                print()
                print("*" * 200)
                for stack in traceback.walk_stack(None):
                    if 'self' in stack[0].f_locals:
                        layer = stack[0].f_locals['self']
                        if type(layer).__name__ in REGISTERED_LIST:
                            print(layer)
                            break
            out = self.obj(self.raw, self.style, *args, **kwargs)
            return out


def replaceTorchAPI(raw, replace):
    raw = Rp(raw, replace)
    return raw


def replaceTorchAPIback(rp):
    return rp.raw


def run():
    F.conv2d = replaceTorchAPI(F.conv2d, _conv2d)
    F.relu = replaceTorchAPI(F.relu, _relu)
    F.leaky_relu = replaceTorchAPI(F.leaky_relu, _leaky_relu)
    F.silu = replaceTorchAPI(F.silu, _silu)
    F.prelu = replaceTorchAPI(F.prelu, _prelu)
    F.max_pool2d = replaceTorchAPI(F.max_pool2d, _max_pool2d)
    F.avg_pool2d = replaceTorchAPI(F.avg_pool2d, _avg_pool2d)
    F.linear = replaceTorchAPI(F.linear, _linear)
    F.adaptive_avg_pool2d = replaceTorchAPI(F.adaptive_avg_pool2d, _adaptive_avg_pool2d)
    F.softmax = replaceTorchAPI(F.softmax, _softmax)
    F.conv_transpose2d = replaceTorchAPI(F.conv_transpose2d, _conv_transpose2d)
    F.pad = replaceTorchAPI(F.pad, _pad)
    F.interpolate = replaceTorchAPI(F.interpolate, _interpolate)
    # torch op
    torch.batch_norm = replaceTorchAPI(torch.batch_norm, _batch_norm)
    torch.sigmoid = replaceTorchAPI(torch.sigmoid, _sigmoid)
    torch.flatten = replaceTorchAPI(torch.flatten, _flatten)
    torch.cat = replaceTorchAPI(torch.cat, _cat)
    torch.instance_norm = replaceTorchAPI(torch.instance_norm, _instance_norm)
    torch.topk = replaceTorchAPI(torch.topk, _topk)
    torch.argmax = replaceTorchAPI(torch.argmax, _argmax)
    torch.matmul = replaceTorchAPI(torch.matmul, _matmul)
    torch.div = replaceTorchAPI(torch.div, _div)  # for [TRT] elt layer's kDIV op
    torch.split = replaceTorchAPI(torch.split, _split)  # for [TRT] slice layer
    torch.reshape = replaceTorchAPI(torch.reshape, _reshape)  # instead view for [TRT] shuffle layer

    # Tensor op
    for t in [torch.Tensor]:
        global raw__add__, raw__sub__, raw__mul__, raw__permute__, raw__expand_as__, raw_get_item
        global raw_operation_dict
        # c = a + b
        raw_operation_dict['add'] = t.__add__
        raw__add__ = t.__add__
        t.__add__ = _add
        # c = a - b
        raw_operation_dict['sub'] = t.__sub__
        raw__sub__ = t.__sub__
        t.__sub__ = _sub
        # c = a * b # for [TRT] scale layer
        raw_operation_dict['mul'] = t.__mul__
        raw__mul__ = t.__mul__
        t.__mul__ = _mul
        # b = a[] get_item for slice
        raw_operation_dict['get_item'] = t.__getitem__
        raw_get_item = t.__getitem__
        t.__getitem__ = _get_item

        # view(instead by torch.reshape), permute for [TRT] shuffle layer
        raw_operation_dict['permute'] = t.permute
        raw__permute__ = t.permute
        t.permute = _permute
        # expand_as for [TRT] expand layer
        raw_operation_dict['expand_as'] = t.expand_as
        raw__expand_as__ = t.expand_as
        t.expand_as = _expand_as
    log.clear()
    os.environ['forward'] = 'true'


def runback():
    os.environ['forward'] = 'false'
    # convolution
    F.conv2d = replaceTorchAPIback(F.conv2d)
    # liner
    F.linear = replaceTorchAPIback(F.linear)
    # activation
    F.relu = replaceTorchAPIback(F.relu)
    F.leaky_relu = replaceTorchAPIback(F.leaky_relu)
    F.silu = replaceTorchAPIback(F.silu)
    F.prelu = replaceTorchAPIback(F.prelu)
    # pooling
    F.max_pool2d = replaceTorchAPIback(F.max_pool2d)
    F.avg_pool2d = replaceTorchAPIback(F.avg_pool2d)
    F.adaptive_avg_pool2d = replaceTorchAPIback(F.adaptive_avg_pool2d)

    F.softmax = replaceTorchAPIback(F.softmax)
    F.conv_transpose2d = replaceTorchAPIback(F.conv_transpose2d)
    F.pad = replaceTorchAPIback(F.pad)
    F.interpolate = replaceTorchAPIback(F.interpolate)
    # torch op
    torch.batch_norm = replaceTorchAPIback(torch.batch_norm)
    torch.sigmoid = replaceTorchAPIback(torch.sigmoid)
    torch.flatten = replaceTorchAPIback(torch.flatten)
    torch.cat = replaceTorchAPIback(torch.cat)
    torch.instance_norm = replaceTorchAPIback(torch.instance_norm)
    torch.topk = replaceTorchAPIback(torch.topk)
    torch.argmax = replaceTorchAPIback(torch.argmax)
    torch.matmul = replaceTorchAPIback(torch.matmul)
    torch.div = replaceTorchAPIback(torch.div)  # for [TRT] elt layer's kDIV op
    torch.split = replaceTorchAPIback(torch.split)  # for [TRT] slice layer
    torch.reshape = replaceTorchAPIback(torch.reshape)  # instead view for [TRT] shuffle layer

    # Tensor op
    for t in [torch.Tensor]:
        # c = a + b
        t.__add__ = raw_operation_dict['add']
        t.__sub__ = raw_operation_dict['sub']
        t.__mul__ = raw_operation_dict['mul']
        t.__getitem__ = raw_operation_dict['get_item']

        t.permute = raw_operation_dict['permute']
        t.expand_as = raw_operation_dict['expand_as']

def GetForwardCall(model,input_tensor):
    log.clear()
    log.add_input_blob_id(int(id(input_tensor)))
    run()
    with torch.no_grad():
        o = model(input_tensor)
    runback()
    return log