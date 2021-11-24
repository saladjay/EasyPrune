# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import logging.handlers

from demo.rawapi import *
from demo.replaceapi import *
from demo.common import *
import torch.nn as nn


class testModel(nn.Module):
    def __init__(self):
        super(testModel, self).__init__()
        self.backbone_self = backbone(3, 0.33, 0.5)
        self.neck_self = neck()

    def forward(self, x):
        x, x6, x4 = self.backbone_self(x)
        x = self.neck_self(x, x6, x4)
        return x


class neck(nn.Module):
    def __init__(self):
        super(neck, self).__init__()
        self.conv1 = Conv(512, 256, 1, 1)
        self.upsample1 = nn.Upsample(None, 2, 'nearest')
        self.cat1 = Concat(dimension=1)
        self.csp1 = C3(512, 256, 1, shortcut=False)

        self.conv2 = Conv(256, 128, 1, 1)
        self.upsample2 = nn.Upsample(None, 2, 'nearest')
        self.cat2 = Concat(dimension=1)
        self.csp2 = C3(256, 128, 1, shortcut=False)

        self.conv3 = Conv(128, 128, 3, 2)
        self.cat3 = Concat(dimension=1)
        self.csp3 = C3(256, 256, 1, shortcut=False)

        self.conv4 = Conv(256, 256, 3, 2)
        self.cat4 = Concat(dimension=1)
        self.csp4 = C3(512, 512, 1, shortcut=False)

    def forward(self, x, x_6, x_4):
        x_10 = self.conv1(x)  # 10 512
        # print('x_10:', x_10.shape)
        x_11 = self.upsample1(x_10)  # 11
        # print('x_11:', x_11.shape)
        x_12 = self.cat1([x_11, x_6])  # 12 512+512
        # print('x_12:', x_12.shape)
        x_13 = self.csp1(x_12)  # 13
        # print('x_13:', x_13.shape)

        x_14 = self.conv2(x_13)  # 14
        # print('x_14:', x_14.shape)
        x_15 = self.upsample2(x_14)  # 15
        # print('x_15:', x_15.shape)
        x_16 = self.cat2([x_15, x_4])  # 16 256+256
        # print('x_16:', x_16.shape)
        x_17 = self.csp2(x_16)  # 17
        # print('x_17:', x_17.shape)

        x_18 = self.conv3(x_17)  # 18
        # print('x_18:', x_18.shape)
        x_19 = self.cat3([x_18, x_14])  # 19 256+256
        # print('x_19:', x_19.shape)
        x_20 = self.csp3(x_19)  # 20
        # print('x_20:', x_20.shape)

        x_21 = self.conv4(x_20)  # 21
        # print('x_21:', x_21.shape)
        x_22 = self.cat4([x_21, x_10])  # 22 512+512
        # print('x_22:', x_22.shape)
        x_23 = self.csp4(x_22)  # 23
        # print('x_23:', x_23.shape)

        return [x_17, x_20, x_23]


class backbone(nn.Module):
    def __init__(self, inp_ch, gd, gw, cp_rate_dict=None):
        super(backbone, self).__init__()
        self.gw = gw
        self.gd = gd
        #             focus, conv, c3, conv, c3, conv, c3, conv, spp,   c3
        self.channels = [64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024]
        self.ori_channels = list(self.channels)
        for index, channel_count in enumerate(self.channels):
            self.channels[index] = get_witdh(channel_count * self.gw, 8)
            self.ori_channels[index] = self.channels[index]

        # if cp_rate_dict is not None:
        #     for index, channel_count in enumerate(self.channels):
        #         self.channels[index] = \
        #             get_prune_channels_keep_channels(cp_rate_dict[channel_count], channel_count, 2)[1]
        #          focus conv c3 conv c3 conv c3 conv spp c3
        self.depths = [1, 1, 3, 1, 9, 1, 9, 1, 1, 3]
        for index, depth_count in enumerate(self.depths):
            self.depths[index] = get_depth(depth_count, self.gd)

        self.focus = Focus(inp_ch, self.channels[0], 3)
        self.conv1 = Conv(self.channels[0], self.channels[1], 3, 2)
        self.csp1 = C3(c1=self.ori_channels[1], c2=self.ori_channels[2],
                       n=self.depths[2], shortcut=True)
        self.conv2 = Conv(self.channels[2], self.channels[3], 3, 2)
        self.csp2 = C3(self.ori_channels[3], self.ori_channels[4],
                       self.depths[4], shortcut=True)
        self.conv3 = Conv(self.channels[4], self.channels[5], 3, 2)
        self.csp3 = C3(self.ori_channels[5], self.ori_channels[6],
                       self.depths[6], shortcut=True)
        self.conv4 = Conv(self.channels[6], self.channels[7], 3, 2)
        self.spp = SPP(self.channels[7], self.channels[8], [5, 9, 13])
        self.csp4 = C3(self.ori_channels[8], self.ori_channels[9],
                       self.depths[9], shortcut=False)

    def forward(self, x):
        # print('inp:', x.shape)
        x_0 = self.focus(x)  # 0
        # print('x_0:', x_0.shape)
        x_1 = self.conv1(x_0)  # 1
        # print('x_1:', x_1.shape)
        x_2 = self.csp1(x_1)  # 2
        # print('x_2:', x_2.shape)
        x_3 = self.conv2(x_2)  # 3
        # print('x_3:', x_3.shape)
        x_4 = self.csp2(x_3)  # 4
        # print('x_4:', x_4.shape)
        x_5 = self.conv3(x_4)  # 5
        # print('x_5:', x_5.shape)
        x_6 = self.csp3(x_5)  # 6
        # print('x_6:', x_6.shape)
        x_7 = self.conv4(x_6)  # 7
        # print('x_7:', x_7.shape)
        x_8 = self.spp(x_7)  # 8
        # print('x_8:', x_8.shape)
        out = self.csp4(x_8)  # 9
        # print('out:', out.shape)
        return [out, x_6, x_4]


class testlist():

    def __getitem__(self, item):
        print(item)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    testlist1 = testlist()
    testlist1[...,::2,::2]
    testlist1[:,:,::2,::2]

    model = testModel()


    from torch.onnx.utils import _trace
    from torch.onnx import OperatorExportTypes

    res = _trace(model, torch.ones(1, 3, 384, 640), OperatorExportTypes.RAW)

    a = 0
    input_tensor = torch.ones(1, 3, 384, 640)
    log.add_input_blob_id(int(id(input_tensor)))
    run()

    print(log.input_blobs)
    with torch.no_grad():
        model(input_tensor)
    runback()

    ori_state_dict = model.state_dict()

    print(log.input_blobs)
    a = 0
    with torch.no_grad():
        model(input_tensor)
    print(log.input_blobs)
    a = 0
    b = log.blobs
    input_set = set()
    import collections
    new_log = collections.OrderedDict()
    for index, (key, value) in enumerate(log.blobs.items()):
        if index == value['index']:
            if index == 0:
                input_set.add(value['output'][0])
                for input_tensor_id in value['input']:
                    input_set.add(input_tensor_id)
                new_log[key] = value
            else:
                flag = False
                for layer_input_id in value['input']:
                    if layer_input_id not in input_set:
                        flag = True
                if not flag:
                    for layer_output_id in value['output']:
                        input_set.add(layer_output_id)
                    new_log[key] = value

    layer_end_keys = ['.weight', '.bias', '.running_mean', '.running_var']
    new_log2 = collections.OrderedDict()
    for key, value in new_log.items():
        if value['type'] == 'conv2d_':
            weight_key = key+'.weight'
            new_weight = log.weight_dict[weight_key] if weight_key in log.weight_dict else None
            flag = False
            if new_weight is not None:
                for key2, old_weight in ori_state_dict.items():
                    if len(old_weight.shape)>0 and old_weight.equal(new_weight):
                        new_log2[key2[:-len('.weight')]] = value
                        flag = True
                        break
            if not flag:
                new_log2[key] = value

        elif value['type'] == 'BN_':
            weight_key = key+'.weight'
            bias_key = key+'.bias'
            running_mean_key = key+'.running_mean'
            running_var_key = key+'.running_var'
            new_weight = log.weight_dict[weight_key] if weight_key in log.weight_dict else None
            new_bias = log.weight_dict[bias_key] if bias_key in log.weight_dict else None
            new_running_mean = log.weight_dict[running_mean_key] if running_mean_key in log.weight_dict else None
            new_running_var = log.weight_dict[running_var_key] if running_var_key in log.weight_dict else None

            flag = False
            equal_list = [0, 0, 0, 0]
            if new_weight is not None:
                for key2, old_weight in ori_state_dict.items():
                    if len(old_weight.shape) == 0:
                        continue
                    if old_weight.equal(new_weight):
                        equal_list[0] = 1
                    if old_weight.equal(new_bias):
                        equal_list[1] = 1
                    if old_weight.equal(new_running_mean):
                        equal_list[2] = 1
                    if old_weight.equal(new_running_var):
                        equal_list[3] = 1
                    if sum(equal_list) == 4:
                        for end_key in layer_end_keys:
                            if key2.endswith(end_key):
                                new_log2[key2[:-len(end_key)]] = value

        else:
            new_log2[key] = value
    a=0