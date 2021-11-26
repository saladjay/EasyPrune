import os

import torch
import numpy as np
import torch.nn as nn
import math
import json


def build_compress_dict_by_epoch(base_rate, channel_min_rate, update_step, max_epoch):
    channels = list(channel_min_rate.keys())
    compress_dict = {}
    e = 0
    while e <= max_epoch:
        ue = e // update_step
        new_rate_dit = {}
        for index, key in enumerate(channels):
            rate = 1 - base_rate * max(0, ue + index - len(channels) + 1)
            new_rate_dit[key] = max(rate, channel_min_rate[key])
        compress_dict[e] = new_rate_dit
        e = e + 1
    return compress_dict


def get_special_prune_rate(base_rate, channel_min_rate, channel_num_list, channel_nums,
                           update_epoch_step, max_epoch, epoch,
                           update_layer_step, max_layer, layer_index):
    layer_index = min(max_layer, layer_index)
    epoch = min(max_epoch, epoch)
    ue = epoch // update_epoch_step
    ul = layer_index // update_layer_step + ue - max_layer
    index = channel_num_list.index(channel_nums)
    prune_rate = 1.0 - base_rate * max(0, ue + ul + index - len(channel_num_list) + 1)
    return max(channel_min_rate, prune_rate)


def get_prune_channels_keep_channels(compress_rate, input_channel, divisor):
    filter_pruned_num = int(input_channel * (1 - compress_rate))
    filter_pruned_num = math.ceil(filter_pruned_num / divisor) * divisor
    return filter_pruned_num, input_channel - filter_pruned_num


def convert2tensor(x):
    x = torch.FloatTensor(x)
    return x


class Mask:
    """
    the soft prune filter mask generator
    can prune conv and fc

    how to use, please follow the step

    before training:
        prune_assistant = PruneAssistant(model)
        rate_dict = {18: 1, 32: 1, 64: 0.875, 128: 0.75, 256: 0.625, 512: 0.625, 1024: 0.625}
        model_layer_dict = prune_assistant.model_layer_dict
        model_layer_prune = prune_assistant.model_layer_prune_rate_dict
        # the dict of final prune rate
        for k,v in model_layer_dict.items():
            shape = getattr(v,'weight').shape if hasattr(v,'weight') else None
            if shape and isinstance(v,nn.Conv2d):
                model_layer_prune[k] = rate_dict[shape[0]]

        Mask = Mask(model,model_layer_prune) #initialize mask
        Mask.init_length() #record layer weight size and length

    during training:
        ... # end of training epoch
        Mask.model = model
        Mask.if_zero()
        Mask.init_mask(True,epoch)
        Mask.do_mask()
        Mask.if_zero()
        model = Mask.model
        ... # other codes
    """

    def __init__(self, model, prune_rate_target, channel_divisor=8, base_rate=1 / 64, update_step=4,
                 max_epoch=150, channel_num_list=None, debug=False, prune_log_dir='./prune_log/'):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.debug = debug
        self.channel_divisor = channel_divisor
        self.compress_rate_dict = prune_rate_target
        self.prune_rate_target_dict = prune_rate_target
        self.prune_layer = set()
        for name, prune_rate in prune_rate_target.items():
            if prune_rate < 1.0:
                self.prune_layer.add(name)
        self.base_rate = base_rate
        self.update_epoch = update_step
        self.update_layer = 1
        self.max_epoch = max_epoch
        self.max_layer = len(prune_rate_target)
        self.channel_num_list = channel_num_list if channel_num_list else [18, 32, 64, 128, 256, 512, 1024, 2048]
        self.prune_log_dir = prune_log_dir
        if prune_log_dir:
            if not os.path.exists(prune_log_dir):
                os.makedirs(prune_log_dir)

            with open(os.path.join(prune_log_dir, 'min_rate.json'), 'w') as min_rate_filestream:
                print(json.dumps(prune_rate_target), file=min_rate_filestream)

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        return self._normalize_codebook(weight_torch, compress_rate, length)

    def _normalize_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:  # for conv
            filter_pruned_num = get_prune_channels_keep_channels(compress_rate, weight_torch.size()[0], 2)[0]
            if self.debug:
                print(f'filter_pruned_num:{filter_pruned_num}')
                print(f'weight_torch shape:{weight_torch.shape}')
                print(f'length:{length}')
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            if self.debug:
                print(f'norm2 shape :{norm2.shape}')
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
        elif len(weight_torch.size()) == 2:
            weight_torch = weight_torch.view(weight_torch.size()[0], weight_torch.size()[1], 1, 1)
            codebook = self.get_filter_codebook(weight_torch, compress_rate, length)
        else:
            pass
        return codebook

    def init_length(self):
        for name, module in self.model.named_modules():
            if name in self.prune_layer:
                self.model_size[name] = (module.weight.size(), module.weight.numel())

    def _init_rate(self, epoch):
        """
        initialize prune rate of per layer in model, the prune rate will be set by layer index and current epoch number
        :param epoch: current epoch number during training
        :return: None
        """
        layer_index = 0
        for index, (name, module) in enumerate(self.model.named_modules()):
            if name in self.prune_layer:
                self.mask_index.append(index)
                min_rate = self.prune_rate_target_dict[name]
                output_channel_num = 0
                if len(module.weight.size()) == 4:  # or isinstance(module, nn.Conv)
                    output_channel_num = module.weight.size()[0]
                elif len(module.weight.size()) == 2:  # or isinstance(module, nn.Linear)
                    output_channel_num = module.weight.size()[1]
                assert output_channel_num != 0, f'{name} layer output channel is zero'
                self.compress_rate[name] = get_special_prune_rate(self.base_rate, min_rate, self.channel_num_list,
                                                                  output_channel_num, self.update_epoch, self.max_epoch,
                                                                  epoch, self.update_layer,
                                                                  len(self.prune_layer), layer_index)
                layer_index += 1
        if self.prune_log_dir:
            if not os.path.exists(self.prune_log_dir):
                os.makedirs(self.prune_log_dir)
            with open(os.path.join(self.prune_log_dir, f'e{epoch}.json'), 'w') as epoch_prune_rate_filestream:
                print(json.dumps(self.compress_rate), file=epoch_prune_rate_filestream)

    def init_mask(self, use_cuda, epoch=-1):
        self._init_rate(epoch)
        for index, (name, module) in enumerate(self.model.named_modules()):
            if name in self.prune_layer:
                self.mat[name] = self.get_filter_codebook(module.weight.data, self.compress_rate[name],
                                                          self.model_size[name][1])
                self.mat[name] = convert2tensor(self.mat[name])
                if use_cuda:
                    self.mat[name] = self.mat[name].cuda()

    def do_mask(self):
        for name, module in self.model.named_modules():
            if name in self.prune_layer:
                weight = module.weight.data.view(self.model_size[name][1])
                pruneWeight = weight * self.mat[name]
                module.weight.data = pruneWeight.view(self.model_size[name][0])

    def if_zero(self):
        for name, module in self.model.named_modules():
            if name in self.prune_layer:
                weight = module.weight.data.clone().detach().view(self.model_size[name][1]).cpu().numpy()
                # print("layer: %s, number of nonzero weight is %d, zero is %d" % (
                #     name, np.count_nonzero(weight), len(weight) - np.count_nonzero(weight)))


if __name__ == '__main__':

    import json

    with open('../prunelog/e0.json', 'r') as f:
        b = json.loads(f.readline())
    print(len(b))
    pr_list = {}
    for i in range(len(b)):
        for o in [18, 32, 64, 128, 256, 512, 1024]:
            pr = get_special_prune_rate(1 / 64, 0.75, [18, 32, 64, 128, 256, 512, 1024],
                                        o, 4, 150,
                                        150, 1,
                                        len(b), i)
            pr_list[f'{i}_{o}'] = pr
    print(pr_list)

    # m = Mask(model)
    #
    # # 通过Mask对象将model进行软剪支
    # m.model = model  #################################################
    # m.if_zero()  #################################################326
    # m.init_mask(1, 3, 30, True)  #################################################326
    # m.do_mask()  #################################################
    # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    # m.if_zero()  #################################################326
    # model = m.model
