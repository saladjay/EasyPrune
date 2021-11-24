import collections
import os
import string


class forwardlog(object):
    def __init__(self):
        self.layers = {}
        self.weight_dict = collections.OrderedDict()
        self.blobs = collections.OrderedDict()
        self.detail_layers = {}
        self.blob_ids = set()
        self.input_blobs = set()

    def add_blob(self, typename, inputs, outputs):
        if os.environ['forward'] == 'true':
            if typename in self.layers:
                return self.layers[typename]
            inputs = inputs if isinstance(inputs, list) or isinstance(inputs, tuple) else [inputs, ]
            outputs = outputs if isinstance(outputs, list) or isinstance(outputs, tuple) else [outputs, ]
            input_ids = [int(id(input_tensor)) for input_tensor in inputs]
            output_ids = [int(id(output_tensor)) for output_tensor in outputs]

            for input_tensor_id in input_ids:
                if input_tensor_id not in self.input_blobs:
                    return None
            for output_tensor_id in output_ids:
                self.input_blobs.add(output_tensor_id)

            blob_dict = {
                'input': input_ids,
                'input_tensor': inputs,
                'type': typename,
                'output': output_ids,
                'output_tensor': outputs,
                'index': len(self.blobs),
            }

            if typename not in self.detail_layers.keys():
                self.detail_layers[typename] = 0
            else:
                self.detail_layers[typename] += 1
            typename = '{}{}'.format(typename, self.detail_layers[typename])

            self.blobs[typename] = blob_dict
            self.layers[typename] = typename
            return self.layers[typename]

    def add_weight(self, name, weight):
        self.weight_dict[name] = weight

    def clear(self):
        self.layers.clear()
        self.weight_dict.clear()
        self.blobs.clear()
        self.detail_layers.clear()
        self.blob_ids.clear()
        self.input_blobs.clear()

    # 增加这个函数是为了，避免替换tensor操作类函数（eg:__add__,__getitem__）被其他函数调用时，添加与模型forward无关的layer
    # 假如模型有多个input，应该分多次将这些tensor的地址加入到iput_blobs里，eg: int(id(x))
    def add_input_blob_id(self, x):
        self.input_blobs.add(x)

    # 去除无用的layer
    def remove_uesless_blob(self):
        input_set = set()
        new_blobs = collections.OrderedDict()
        for index, (key, value) in enumerate(self.blobs.items()):
            if index == value['index']:
                if index == 0:
                    input_set.add(value['output'][0])
                    for input_tensor_id in value['input']:
                        input_set.add(input_tensor_id)
                    new_blobs[key] = value
                else:
                    flag = False
                    for layer_input_id in value['input']:
                        if layer_input_id not in input_set:
                            flag = True
                    if not flag:
                        for layer_output_id in value['output']:
                            input_set.add(layer_output_id)
                        new_blobs[key] = value
        self.blobs = new_blobs

    def rename_blobs(self, state_dict):
        layer_end_keys = ['.weight', '.bias', '.running_mean', '.running_var']
        new_blobs = collections.OrderedDict()
        for key, value in self.blobs.items():
            if value['type'] == 'conv2d_':
                weight_key = key + '.weight'
                new_weight = self.weight_dict[weight_key] if weight_key in self.weight_dict else None
                flag = False
                if new_weight is not None:
                    for key2, old_weight in state_dict.items():
                        if len(old_weight.shape) > 0 and old_weight.equal(new_weight):
                            new_blobs[key2[:-len('.weight')]] = value
                            flag = True
                            break
                if not flag:
                    new_blobs[key] = value

            elif value['type'] == 'BN_':
                weight_key = key + '.weight'
                bias_key = key + '.bias'
                running_mean_key = key + '.running_mean'
                running_var_key = key + '.running_var'
                new_weight = self.weight_dict[weight_key] if weight_key in self.weight_dict else None
                new_bias = self.weight_dict[bias_key] if bias_key in self.weight_dict else None
                new_running_mean = self.weight_dict[running_mean_key] if running_mean_key in self.weight_dict else None
                new_running_var = self.weight_dict[running_var_key] if running_var_key in self.weight_dict else None

                equal_list = [0, 0, 0, 0]
                if new_weight is not None:
                    for key2, old_weight in state_dict.items():
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
                                    new_blobs[key2[:-len(end_key)]] = value

            else:
                new_blobs[key] = value
        self.blobs = new_blobs

    def getlayername(self, index):
        for i, (key, blob) in enumerate(self.blobs):
            assert i < blob['index'], 'blobs has error'
            if blob['output'][0] == index:
                return key
        return None

    def to_csv(self, path):
        """
        :param path: csv file path
        :return: dataframe
        """
        import pandas as pd
        layer_names = []
        inputs = []
        outputs = []
        layer_indexes = []
        layer_types = []
        input_shapes = []
        output_shapes = []
        weight_shapes = []
        for i, (key, blob) in enumerate(self.blobs):
            assert i < blob['index'], 'blobs has error'
            layer_names.append(key)

            input_names = []
            for input_id in blob['input']:
                name = self.getlayername(input_id)
                input_names.append(name if name is not None else input_id)
            inputs.append(input_names)

            outputs.append([f'{key}{oi}' for oi in range(len(blob['output']))])

            layer_indexes.append(i)

            layer_types.append(blob['type'])

            input_shapes.append([list(t.shape) for t in blob['input_tensor']])

            output_shapes.append(([list(t.shape) for t in blob['output_tensor']]))

            w = self.weight_dict.get(f'{key}.weight')
            weight_shapes.append(list(w.shape) if w is not None else None)
        csv = pd.DataFrame()
        csv['name'] = layer_names
        csv['input'] = inputs
        csv['output'] = outputs
        csv['index'] = layer_indexes
        csv['type'] = layer_types
        csv['input_shape'] = input_shapes
        csv['output_shape'] = output_shapes
        csv['weight_shape'] = weight_shapes
        csv.to_csv(path)
        return csv
