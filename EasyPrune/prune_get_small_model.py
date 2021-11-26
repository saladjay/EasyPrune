import torch
import torch.nn as nn
import json
import collections
import numpy as np
from copy import deepcopy

from EasyPrune.prune_utils import *
from EasyPrune.prune_common import *
from EasyPrune.forwardlog import *
from EasyPrune.replaceapi import *


def get_special_module(model, name):
    """
    find the special module

    Parameters:
        model: the model contain the special module
        name: the name of the special module

    Returns:
        the module with that name
        the layer contain the module
        the name of the module in the layer
    """
    layers_name = name.split('.')
    if len(layers_name) > 1:
        assert hasattr(model, layers_name[0]), 'model does not contain this attribute'
        return get_special_module(getattr(model, layers_name[0]), '.'.join(layers_name[1:]))
    else:
        return (getattr(model, layers_name[0]), model, name) if hasattr(model, layers_name[0]) else (None, None, name)


def set_special_module(model, name, new_module):
    if name.find('.') == -1:
        assert hasattr(model, name), f'{type(model)} does not contain attribute'
        assert new_module is not None, f'new module is None'
        setattr(model, name, new_module)
    else:
        old_module, layer, detailed_name = get_special_module(model, name)
        assert layer is not None, f'{name} is not contained in {type(model)}'
        assert old_module is not None, f'{name} is not contained in {type(model)}'
        setattr(layer, detailed_name, new_module)


def load_json(json_path):
    assert os.path.exists(json_path), f'{json_path} does not exist'
    with open(json_path, 'r') as json_stream:
        return json.loads(json_stream.read())


def get_weight_shape(weights_dict, layer_name):
    return weights_dict[layer_name].shape


def get_layer_by_output_id(blobs, output_id):
    for blob_name, blob in blobs.items():
        if output_id in blob['output']:
            return blob_name, blob
    return None, None


class small_model_assist(object):
    def __init__(self, relative_blob_types: set[str], ignore_blob_types: set[str]):
        super(small_model_assist, self).__init__()
        self.relative_blob_types = relative_blob_types
        self.ignore_blob_types = ignore_blob_types
        self.relative_blob_types.add('add_')
        self.relative_blob_types.add('cat_')

    def get_relative_input_blobs(self, blobs, blob_name):
        """
        Returns:
            a list of containing input layers, [(name, layer_dict), (name, layer_dict)]
        """
        blob = blobs[blob_name]
        input_blobs = [get_layer_by_output_id(blobs, input_id) for input_id in blob['input']]
        relative_input_layers = []
        for input_blob_name, input_blob in input_blobs:
            if input_blob['type'] in self.ignore_blob_types:
                relative_input_layers += self.get_relative_input_blobs(blobs, input_blob_name)
            elif input_blob['type'] in self.relative_blob_types:
                relative_input_layers.append((input_blob_name, input_blob))
            else:
                print(f'unexpected blob type in blobs:{input_blob["type"]}')
        return relative_input_layers

    def get_relative_input_blobs_expression_tree(self, blobs, blob_name):
        blob_type = blobs[blob_name]['type']
        relative_input_blobs = self.get_relative_input_blobs(blobs, blob_name)
        blob_expression_tree = []
        for index, blob in enumerate(relative_input_blobs):
            if blob[1]['type'] in ['add_', 'cat_']:
                blob_expression_tree.append(self.get_relative_input_blobs_expression_tree(blobs, blob_name))
            else:
                blob_expression_tree.append(blob)
            if index < len(relative_input_blobs) - 1:
                blob_expression_tree.append(blob_type)
        return blob_expression_tree

    def analysis_blobs_expression_tree_for_blob_order(self, model, expression_tree):
        blob_type = None
        input_blobs = []

        for value in expression_tree:
            if isinstance(value, list):
                input_blobs += self.get_relative_input_blobs(model, value)

            elif isinstance(value, tuple):
                special_module, special_layer, special_attr_name = get_special_module(model, value[0])
                assert special_module is not None, f'{value[0]} does not exist in model'
                if len(input_blobs) == 0:
                    input_blobs.append(value[0])
                elif len(input_blobs) > 0:
                    if blob_type == 'cat_':
                        input_blobs.append(value[0])
                    elif blob_type == 'add_':
                        pass
            elif isinstance(value, str):
                blob_type = value
        return input_blobs

    def analysis_blobs_expression_tree_for_prune_input_channels(self, model, expression_tree, prune_blob_dict):
        blob_type = None
        blob_output_channels = []
        for value in expression_tree:
            if isinstance(value, list):
                o_channels = self.analysis_blobs_expression_tree_for_prune_input_channels(model, value, prune_blob_dict)
                assert len(blob_output_channels) != 1, ''

                if len(blob_output_channels) == 0:
                    blob_output_channels.append(o_channels)

                if len(blob_output_channels) == 2:
                    if blob_output_channels[-1] == 'cat_':
                        sum_channels = o_channels + blob_output_channels[0]
                        blob_output_channels.clear()
                        blob_output_channels.append(sum_channels)

                    elif blob_output_channels[-1] == 'add_':
                        out_c = max(blob_output_channels[0], o_channels)
                        blob_output_channels.clear()
                        blob_output_channels.append(out_c)

            elif isinstance(value, tuple):
                special_module, special_blob, special_name = get_special_module(model, value[0])
                assert special_module is not None, f'{value[0]} does not exist in model'
                prune_rate = prune_blob_dict[value[0]] if value[0] in prune_blob_dict else 1
                if len(blob_output_channels) == 0:
                    if isinstance(special_module, nn.Conv2d):
                        blob_output_channels.append(special_module.out_channels * prune_rate)
                    elif isinstance(special_module, nn.Linear):
                        pass
                    else:
                        print(f'new type of module to prune: {type(special_module)}')
                        pass
                elif len(blob_output_channels) == 2:
                    if blob_output_channels[-1] == 'cat_':
                        sum_channels = special_module.out_channels * prune_rate + blob_output_channels[0]
                        blob_output_channels.clear()
                        blob_output_channels.append(sum_channels)
                        # to do :
                        # cat operation with more detail: forget what is the detail

                    elif blob_output_channels[-1] == 'add_':
                        out_c = max(int(blob_output_channels[0]), int(special_module.out_channels * prune_rate))
                        blob_output_channels.clear()
                        blob_output_channels.append(out_c)

                elif isinstance(value, str):
                    assert len(blob_output_channels) == 1, ''
                    blob_output_channels.append(value)

            assert len(blob_output_channels) == 1, f'{blob_output_channels} origin expression is {expression_tree}'
            return blob_output_channels[0]


def get_small_model(ori_model, json_path, input_tensor):
    layer_prune_rate_dict = load_json(json_path)
    forward_log = get_forward_log(ori_model, input_tensor)
    prune_assist = PruneAssistant(ori_model)
    ori_state_dict = ori_model.state_dict()
    weight_dict = forward_log.weight_dict
    model_assist = small_model_assist(set('conv2d_'), set('interpolate_', 'max_pool2d_'))
    convert_dict = {}
    small_model = deepcopy(ori_model)

    for k1, v1 in weight_dict.items():
        for k2, v2 in ori_state_dict.items():
            if len(v1.shape) > 0 and len(v2.shape) > 0 and v1.equal(v2):
                convert_dict[k1] = k2

    blobs_dict = forward_log.blobs
    final_dict = collections.OrderedDict()

    for k2, v2 in blobs_dict.items():
        flag = False
        for k1, v1 in convert_dict.items():
            if k1.startswith(k2):
                len_suffix = len(k1) - len(k2)
                final_dict[v1[:-len_suffix]] = v2
                flag = True
                break
        if not flag:
            final_dict[k2] = v2

    for module_name, module in ori_model.named_modules():
        if len(list(module.children())) == 0:
            if isinstance(module, nn.Conv2d):
                if module_name in layer_prune_rate_dict:
                    # find input layer
                    input_blobs = model_assist.get_relative_input_blobs(blobs_dict, module_name)

                    ori_input_channels = module.in_channels
                    ori_output_channels = module.out_channels

                    d = {'name': module_name, 'prune': layer_prune_rate_dict[module_name],
                         'input_c': ori_input_channels, 'output_c': ori_output_channels,
                         'input_tensor': [(n, b['output_tensor_shape'][0]) for n, b in input_blobs]}
                    # calculate this convolution input channels and output channels after pruning
                    prune_output_c = ori_output_channels * d['prune']
                    prune_input_c = model_assist.analysis_blobs_expression_tree_for_prune_input_channels(final_dict,
                                                                                                         module_name,
                                                                                                         ori_model,
                                                                                                         layer_prune_rate_dict)
                    d['prune_oc'] = prune_output_c
                    d['prune_ic'] = prune_input_c
                    # create a new convolution2d module to replace the old one
                    param_d = prune_assist.model_blob_parameters[module_name]
                    param_d['in_channels'] = int(prune_input_c)
                    param_d['out_channels'] = int(prune_output_c)
                    new_conv = nn.Conv2d(**param_d)
                    # replace
                    set_special_module(small_model, module_name, new_conv)

                else:
                    input_blobs = model_assist.get_relative_input_blobs(final_dict, module_name)
                    if sum([n in blobs_dict for n, b in input_blobs]):
                        ori_input_channels = module.in_channels
                        ori_output_channels = module.out_channels
                        d = {'name': module_name, 'prune': 1,
                             'input_c': ori_input_channels, 'output_c': ori_output_channels,
                             'input_tensor': [(n, b['output_tensor_shape'][0]) for n, b in input_blobs]}
                        prune_output_c = ori_output_channels * d['prune']
                        prune_input_c = sum(
                            [s[1] * (layer_prune_rate_dict[n] if n in layer_prune_rate_dict else 1) for n, s in
                             d['input_tensor']]
                        )
                        d['prune_oc'] = prune_output_c
                        d['prune_ic'] = prune_input_c

                        param_d = prune_assist.model_blob_parameters[module_name]
                        param_d['in_channels'] = int(prune_input_c)
                        param_d['out_channels'] = int(prune_output_c)
                        set_special_module(small_model, module_name, nn.Conv2d(**param_d))

            if isinstance(module, nn.BatchNorm2d):
                input_blob = model_assist.get_relative_input_blobs(final_dict, module_name)
                # usually the module in front of batch normalization is the convolution2d
                assert len(input_blob) == 1, ''

                d = {'name': module_name,
                     'ori_num_features': module.num_features,
                     'prune': layer_prune_rate_dict[input_blob[0][0]] if input_blob[0][
                                                                             0] in layer_prune_rate_dict else 1}
                d['prune_num_features'] = module.num_features * d['prune']

                param_d = prune_assist.model_blob_parameters[module_name]
                param_d['num_features'] = int(d['prune_num_features'])
                set_special_module(small_model, module_name, nn.BatchNorm2d(**param_d))

    return small_model, final_dict


def get_keep_index_of_weight(tensor: torch.Tensor):
    # now only support convolution weight
    assert len(tensor.shape) == 4, ''
    size_0 = tensor.size()[0]
    size_1 = tensor.size()[1] * tensor.size()[2] * tensor.size()[3]
    tensor_resize = tensor.view(size_0, -1)
    # indicator: if the channel contain all zeros
    channel_if_zero = np.zeros(size_0)
    for x in range(0, size_0, 1):
        channel_if_zero[x] = np.count_nonzero(tensor_resize[x].cpu().numpy()) != 0

    indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])

    zeros = (channel_if_zero == 0).nonzero()[0]
    indices_zero = torch.LongTensor(zeros) if zeros.size > 0 else []

    return indices_zero, indices_nonzero


if __name__ == '__main__':
    layer_dict = loadJson('../prunelog/e199.json')
    model = Model('./yolov5s_package_detection.yaml', ch=3, nc=1, anchors=None)
    model.train()
    o1 = model(torch.ones(1, 3, 384, 640))
    model.load_state_dict(torch.load('../runs/train/new-asfp3/weights/precision.pt')['model'].state_dict())
    model = torch.load('../runs/train/new-asfp3/weights/precision.pt')['model']
    model.float()
    o_state_dict = model.state_dict()
    input_tensor = torch.ones(1, 3, 384, 640)
    for name,weight in o_state_dict.items():
        print(name)
    prune_model, final_dict = GetPruneModel(model, '../prunelog/e194.json', input_tensor)
    prune_model.train()
    print('*'*20)
    for name, module in prune_model.named_modules():
        if len(list(module.children())) == 0:
            print(name, module)
    # o = prune_model(torch.ones(1, 3, 384, 640))
    # prune_model.cuda()
    # o2 = prune_model(torch.ones(1, 3, 384, 640).cuda())
    print('='*30)
    keep_dict = {}
    for name,module in model.named_modules():
        if len(list(module.children())) == 0:
            if isinstance(module,nn.Conv2d):
                weight = module.weight.data
                index_zeros, index_keeps = check_channel(weight)
                prune_module,_,_ = GetDetailedModule(prune_model, name)
                assert isinstance(prune_module, nn.Conv2d), ''

                print(name, len(index_keeps), prune_module.out_channels)
                assert len(index_keeps), prune_module.out_channels
                keep_dict[name] = index_keeps

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            if isinstance(module, nn.BatchNorm2d):
                conv_layer = calculateInputLayers(final_dict,name)
                assert len(conv_layer) == 1, ''
                conv_name = conv_layer[0][0]
                keep_dict[name] = keep_dict[conv_name]

    end_keyword = ['.weight', '.bias', '.running_mean', '.running_var']
    dst_model = deepcopy(prune_model)
    ori_model = deepcopy(model)
    indice_dict1 = deepcopy(keep_dict)
    indice_dict = {}
    for k,v in indice_dict1.items():
        for subfix in end_keyword:
            indice_dict[k+subfix] = v
        indice_dict[k] = v

    new_indice_dict = deepcopy(indice_dict)

    dst_state_dict = dst_model.state_dict()
    ori_state_dict = ori_model.state_dict()
    for ori_key, ori_value in ori_state_dict.items():
        dst_value = dst_state_dict[ori_key]
        ori_shape = ori_value.shape
        dst_shape = dst_value.shape

        if len(ori_shape) == 4:
            if ori_shape == dst_shape:
                dst_state_dict[ori_key] = ori_value

            # 剪枝第0维度,也就是输出维度.
            elif ori_shape[0] > dst_shape[0] and ori_shape[1] == dst_shape[1]:
                select_index = indice_dict[ori_key]
                if len(select_index) != dst_shape[0]:
                    print(f'input error :{len(select_index)},{dst_shape}')
                pruneWeight = ori_value.detach().cpu()
                pruneWeight = torch.index_select(pruneWeight, 0, select_index)
                dst_state_dict[ori_key] = pruneWeight

            # 剪枝第1维度，也就是输入维度
            elif ori_shape[0] == dst_shape[0] and ori_shape[1] > dst_shape[1]:
                for end_key in end_keyword:
                    if ori_key.endswith(end_key):
                        keyword = ori_key[:-len(end_key)]
                        print('*' * 20)
                        print(keyword, ori_key)
                        input_syntax = layerSyntaxAnalysis(final_dict, keyword)
                        layers = layerSyntaxExecuteForInputIndex(ori_model, input_syntax)
                        assert (keyword != '')
                        input_layer_tensors = []
                        last_channel_count = 0
                        for i, item in enumerate(layers):
                            key = item
                            input_layer_tensor = new_indice_dict[key].detach().clone()
                            input_layer_tensor = input_layer_tensor + last_channel_count
                            last_channel_count = ori_state_dict[key + '.weight'].shape[0]

                            input_layer_tensors.append(input_layer_tensor)
                        select_index = input_layer_tensors[0] if len(input_layer_tensors) == 1 else torch.cat(
                            input_layer_tensors, dim=0)
                        #                     print(select_index)
                        print('*' * 20)
                        print()
                        if len(select_index) != dst_shape[1]:
                            print(f'output error :{len(select_index)},{dst_shape}')
                        pruneWeight = ori_value.detach().cpu()
                        pruneWeight = torch.index_select(pruneWeight, 1, select_index)
                        dst_state_dict[ori_key] = pruneWeight

            elif ori_shape[0] > dst_shape[0] and ori_shape[1] > dst_shape[1]:

                select_index = indice_dict[ori_key]
                if len(select_index) != dst_shape[0]:
                    print(f'output error :{len(select_index)},{dst_shape}')
                pruneWeight = ori_value.detach().cpu()
                pruneWeight = torch.index_select(pruneWeight, 0, select_index)

                keyword = ''
                for end_key in end_keyword:
                    if ori_key.endswith(end_key):
                        keyword = ori_key[:-len(end_key)]
                        print('=' * 20)
                        print(keyword, ori_key)
                        input_syntax = layerSyntaxAnalysis(final_dict, keyword)
                        layers = layerSyntaxExecuteForInputIndex(ori_model, input_syntax)

                        assert (keyword != '')
                        input_layer_tensors = []
                        last_channel_count = 0
                        for i, item in enumerate(layers):
                            key = item
                            input_layer_tensor = new_indice_dict[key].detach().clone()

                            input_layer_tensor = input_layer_tensor + last_channel_count
                            last_channel_count = ori_state_dict[key + '.weight'].shape[0]

                            input_layer_tensors.append(input_layer_tensor)
                        select_index = input_layer_tensors[0] if len(input_layer_tensors) == 1 else torch.cat(
                            input_layer_tensors, dim=0)
                        if len(select_index) != dst_shape[1]:
                            print(f'input error :{len(select_index)},{dst_shape}')
                        #                     print(select_index)
                        print('=' * 20)
                        print()
                        pruneWeight = torch.index_select(pruneWeight, 1, select_index)
                        dst_state_dict[ori_key] = pruneWeight

        elif len(ori_shape) == 1:
            if ori_shape == dst_shape:
                dst_state_dict[ori_key] = ori_value
            elif ori_shape[0] > dst_shape[0]:
                select_index = indice_dict[ori_key]
                if len(select_index) != dst_shape[0]:
                    print(f'input error :{len(select_index)},{dst_shape}')
                pruneWeight = ori_value.detach().cpu()
                #             print(key,pruneWeight,len(pruneWeight))
                pruneWeight = torch.index_select(pruneWeight, 0, select_index)
                dst_state_dict[ori_key] = pruneWeight
    prune_model.load_state_dict(dst_state_dict)
    prune_model.cpu()
    o = prune_model(torch.ones(1,3,384,640))
    ckpt = torch.load('../runs/train/new-asfp3/weights/best.pt')
    ckpt['model'] = prune_model
    ckpt['epoch'] = 0
    torch.save(ckpt, '../runs/train/new-asfp3/weights/prune.pt')
    pass