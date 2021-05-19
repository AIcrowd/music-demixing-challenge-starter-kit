import math
from warnings import warn

import torch
import torch.nn as nn

from lasaft.source_separation.sub_modules.building_blocks import TFC, TDF_f1_to_f2

class TFC_LaSAFT(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor, min_bn_units, bias,
                 activation, condition_dim, num_tdfs, dk):
        super(TFC_LaSAFT, self).__init__()
        import math
        self.dk_sqrt = math.sqrt(dk)
        self.num_tdfs = num_tdfs
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf, activation)
        self.tdfs = TDF_f1_to_f2(gr, f, f * num_tdfs, bn_factor, bias, min_bn_units, activation)
        self.keys = nn.Parameter(torch.randn(dk, num_tdfs), requires_grad=True)
        self.linear_query = nn.Linear(condition_dim, dk)
        self.activation = self.tdfs.tdf[-1]

    def forward(self, x, c):
        x = self.tfc(x)
        return x + self.lasaft(x, c)

    def lasaft(self, x, c):
        query = self.linear_query(c)
        qk = torch.matmul(query, self.keys) / self.dk_sqrt
        value = (self.tdfs(x)).view(list(x.shape)[:-1] + [-1, self.num_tdfs])
        att = qk.softmax(-1)
        return torch.matmul(value, att.unsqueeze(-2).unsqueeze(-3).unsqueeze(-1)).squeeze(-1)


class TFC_LightSAFT(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor, min_bn_units, bias,
                 activation, condition_dim, num_tdfs, dk):
        super(TFC_LightSAFT, self).__init__()
        import math
        self.dk_sqrt = math.sqrt(dk)
        self.num_tdfs = num_tdfs
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf, activation)
        num_bn_features = f // bn_factor
        if num_bn_features < min_bn_units:
            num_bn_features = min_bn_units

        self.LaSAFT_bn_layer = nn.Sequential(
            nn.Linear(f, num_bn_features * num_tdfs, bias),
            nn.BatchNorm2d(gr),
            activation()
        )
        self.LaSIFT = nn.Sequential(
            nn.Linear(num_bn_features, f, bias),
            nn.BatchNorm2d(gr),
            activation()
        )
        self.keys = nn.Parameter(torch.randn(dk, num_tdfs), requires_grad=True)
        self.linear_query = nn.Linear(condition_dim, dk)

    def forward(self, x, c):
        x = self.tfc(x)
        return x + self.lasaft(x, c)

    def lasaft(self, x, c):
        query = self.linear_query(c)
        qk = torch.matmul(query, self.keys) / self.dk_sqrt
        value = (self.LaSAFT_bn_layer(x)).view(list(x.shape)[:-1] + [self.num_tdfs, -1])
        att = qk.softmax(-1)

        lasaft_features = torch.matmul(att.unsqueeze(-2).unsqueeze(-3).unsqueeze(-2), value).squeeze(-2)
        return self.LaSIFT(lasaft_features)

