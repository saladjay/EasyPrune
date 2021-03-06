import torch
import torch.nn as nn
import collections
import inspect


class PruneAssistant:
    def __init__(self,  model):
        super(PruneAssistant, self).__init__()
        self.model_layer_dict = collections.OrderedDict()
        self.model_layer_prune_rate_dict = collections.OrderedDict()
        self.model_layer_parameters = {}
        self.layer_parameter_dict = {}
        self.model = model
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential) or len(list(module.children())) > 0:
                continue
            self.model_layer_dict[name] = module
            self.model_layer_prune_rate_dict[name] = 1.0
            self.model_layer_parameters[name] = {}
            if type(module) not in self.layer_parameter_dict.keys():
                self.layer_parameter_dict[type(module)] = inspect.signature(type(module)).parameters
            for k, v in module.__dict__.items():
                if k in self.layer_parameter_dict[type(module)].keys():
                    self.model_layer_parameters[name][k] = v

    def save(self, path):
        checkpoint = {'model_layer_dict': self.model_layer_dict,
                      'model_layer_prune_rate_dict': self.model_layer_prune_rate_dict,
                      'model_layer_parameters': self.model_layer_parameters,
                      'layer_parameter_dict': self.layer_parameter_dict,
                      'model': self.model}
        torch.save(checkpoint, path)
        del checkpoint

    def load(self, path):
        checkpoint = torch.load(path)
        self.model_layer_dict = checkpoint['model_layer_dict']
        self.model_layer_prune_rate_dict = checkpoint['model_layer_prune_rate_dict']
        self.model_layer_parameters = checkpoint['model_layer_parameters']
        self.layer_parameter_dict = checkpoint['layer_parameter_dict']
        self.model = checkpoint['model']
        del checkpoint

    # def get_small_model(self):


if __name__ == '__main__':
    pass