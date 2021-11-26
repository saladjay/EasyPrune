import torch
import torch.nn as nn
import collections
import inspect

class PruneAssistant():
    def __init__(self, model):
        super(PruneAssistant, self).__init__()
        self.model_blob_dict = collections.OrderedDict()
        self.model_blob_prune_rate_dict = collections.OrderedDict()
        self.model_blob_parameters = {}
        self.model_blob_parameter_dict = {}
        self.model = model
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Sequential) or len(list(module.children)) > 0:
                continue
            self.model_blob_dict[module_name] = module
            self.model_blob_prune_rate_dict[module_name] = 1.0
            self.model_blob_parameters[module_name] = {}
            if type(module) not in self.model_blob_parameter_dict.keys():
                self.model_blob_parameter_dict[type(module)] = inspect.signature(type(module)).parameters
            for k, v in module.__dict__.items():
                if k in self.model_blob_parameter_dict[type(module)].keys():
                    self.model_blob_parameters[module_name][k] = v

    def save(self, path):
        ckpt = {'model_layer_dict': self.model_blob_dict,
                'model_layer_prune_rate_dict': self.model_blob_prune_rate_dict,
                'model_layer_parameters': self.model_blob_parameters,
                'layer_parameter_dict': self.model_blob_parameter_dict,
                'model': model}
        torch.save(ckpt, path)
        del ckpt

    def load(self, path):
        ckpt = torch.load(path)
        self.model_blob_dict = ckpt['model_layer_dict']
        self.model_blob_prune_rate_dict = ckpt['model_layer_prune_rate_dict']
        self.model_blob_parameters = ckpt['model_layer_parameters']
        self.model_blob_parameter_dict = ckpt['layer_parameter_dict']
        self.model = ckpt['model']
        del ckpt

    # def get_small_model(self):



if __name__ == '__main__':
    import torchvision
    layer_parameter_dict = {}
    model = torchvision.models.resnet50(pretrained=None)
    model_layer_dict = collections.OrderedDict()
    model_layer_prune_rate = collections.OrderedDict()
    model_layer_parameters = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            continue
        if len(list(module.children())) > 0:
            continue
        model_layer_dict[name] = module
        model_layer_prune_rate[name] = 1.0
        model_layer_parameters[name] = {}
        if type(module) not in layer_parameter_dict.keys():
            layer_parameter_dict[type(module)] = inspect.signature(type(module)).parameters

        for k,v in module.__dict__.items():
            if k in layer_parameter_dict[type(module)].keys():
                model_layer_parameters[name][k] = v

    a = 0