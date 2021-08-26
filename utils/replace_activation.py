import torch
import torch.nn.functional as F
from torch import nn


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input):
        return input * (torch.tanh(F.softplus(input)))


class TanhExp(nn.Module):
    """
    Xinyu Liu, Xiaoguang Di
    TanhExp: A Smooth Activation Function
    with High Convergence Speed for
    Lightweight Neural Networks
    https://arxiv.org/pdf/2003.09855v1.pdf
    """

    def __init__(self):
        super(TanhExp, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.exp(x))


def replace_activations(model, existing_layer, new_layer):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activations(
                module, existing_layer, new_layer
            )

        if type(module) == existing_layer:
            layer_old = module
            layer_new = new_layer
            model._modules[name] = layer_new
    return model
