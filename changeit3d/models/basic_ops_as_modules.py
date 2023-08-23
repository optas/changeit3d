"""
Basic torch tensor manipulations captured via nn.Modules

Originally created at 9/20/20, for Python 3.x
2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""
import torch
from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
      
    def forward(self, x):
        return torch.sigmoid(x)
    

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
      
    def forward(self, x):
        return torch.relu(x)