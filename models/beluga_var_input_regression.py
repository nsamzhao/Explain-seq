"""
An example model that has double the number of convolutional layers
that DeepSEA (Zhou & Troyanskaya, 2015) has. Otherwise, the architecture
is identical to DeepSEA.

When making a model architecture file of your own, please review this
file in its entirety. In addition to the model class, Selene expects
that `criterion` and `get_optimizer(lr)` are also specified in this file.
"""
import numpy as np
import torch
import torch.nn as nn


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

    
class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

def conv_output_shape(h_w, kernel_size=1, stride=(1, 1), pad=0, dilation=1):
    #by Duane Nielsen
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride[0]) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride[1]) + 1)
    return h, w

class Beluga(nn.Module):
    def __init__(self, sequence_length, n_targets):
        super(Beluga, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        # calculate conv output shape
        out=conv_output_shape((1,sequence_length), kernel_size=(1,8))
        out1=conv_output_shape(out, kernel_size=(1,8))
        out2=conv_output_shape(out1, kernel_size=(1, 4), stride=(1, 4))
        out3=conv_output_shape(out2, kernel_size=(1, 8))
        out4=conv_output_shape(out3, kernel_size=(1, 8))
        out5=conv_output_shape(out4, kernel_size=(1, 4), stride=(1, 4))
        out6=conv_output_shape(out5, kernel_size=(1, 8))
        out7=conv_output_shape(out6, kernel_size=(1, 8))
        self.model = nn.Sequential(
            nn.Sequential(
                Lambda(lambda x: x.reshape(-1,4,1,sequence_length)), #added, and changed from view to reshape
                nn.Conv2d(4,320,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, pool_kernel_size),(1, pool_kernel_size)),
                nn.Conv2d(320,480,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, pool_kernel_size),(1, pool_kernel_size)),
                nn.Conv2d(480,640,(1, conv_kernel_size)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, conv_kernel_size)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.reshape(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.reshape(1,-1) if 1==len(x.size()) else x ),nn.Linear(640*(out7[0]*out7[1]),2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.reshape(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,n_targets)),
            ),
        )

    def forward(self, x):
        return self.model(x)


def criterion():
    """
    Specify the appropriate loss function (criterion) for this
    model.

    Returns
    -------
    torch.nn._Loss
    """
    return nn.MSELoss()

def get_optimizer(lr):
    """
    Specify an optimizer and its parameters.

    Returns
    -------
    tuple(torch.optim.Optimizer, dict)
        The optimizer class and the dictionary of kwargs that should
        be passed in to the optimizer constructor.

    """
#    return (torch.optim.SGD,
#            {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
    return (torch.optim.Adam,
            {"lr": lr}) # default lr is 0.001
