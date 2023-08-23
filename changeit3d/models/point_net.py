""" Point-Net.
Originally created at 5/22/20, for Python 3.x
2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, init_feat_dim, conv_dims=[64, 64, 128, 128, 1024],
                 b_norm=True, kernel_size=1, pooling='max',
                 fc_neurons=None, close_with_bias=True):
        super(PointNet, self).__init__()
        ops = []
        previous_dim = init_feat_dim
        for dim in conv_dims:
            ops.append(nn.Conv1d(previous_dim, dim, kernel_size))

            if b_norm:
                ops.append(nn.BatchNorm1d(dim))

            ops.append(nn.ReLU())
            previous_dim = dim

        self.pre_pooling_feature = nn.Sequential(*ops)
        self.pooling = pooling

        ops = []
        if fc_neurons is not None:
            previous_dim = conv_dims[-1]
            for neurons in fc_neurons[:-1]:
                ops.append(nn.Linear(previous_dim, neurons))
                if b_norm:
                    ops.append(nn.BatchNorm1d(neurons))
                ops.append(nn.ReLU())
                previous_dim = neurons

            ops.append(nn.Linear(previous_dim, fc_neurons[-1], bias=close_with_bias))
            self.fc_feature = nn.Sequential(*ops)
        else:
            self.fc_feature = None

    def __call__(self, x):
        out = self.pre_pooling_feature(x)

        if self.pooling is not None:
            if self.pooling == 'max':
                out = F.max_pool1d(out, kernel_size=out.shape[-1]).squeeze_(-1)

        if self.fc_feature is not None:
            out = self.fc_feature(out)
        return out
