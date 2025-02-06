# -*- coding: utf-8 -*-
# @Time    : 4/6/2023 8:43 PM
# @Author  : Gang Qu
# @FileName: GCN.py

import dgl
import torch
import torch.nn as nn
from dgl.nn import GraphConv
from torch.nn import functional as F


class GCN(nn.Module):
    def __init__(self, net_params):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()

        in_channels = net_params['in_channels']
        hidden_channels = net_params['hidden_channels']
        out_channels = net_params['out_channels']
        num_layers = net_params['num_layers']
        dropout = net_params['dropout']
        self.additional_feature = net_params['additional_feature']
        self.readout_type = net_params['readout']
        if net_params['activation'] == "elu":
            activation = F.elu
        else:
            activation = F.relu
        self.layers.append(GraphConv(in_channels, hidden_channels-self.additional_feature, activation=activation, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        for i in range(1, num_layers):
            if i != num_layers - 1:
                self.layers.append(GraphConv(hidden_channels, hidden_channels, activation=activation, allow_zero_in_degree=True))
            else:
                self.layers.append(GraphConv(hidden_channels, out_channels, activation=activation, allow_zero_in_degree=True))
        self.predict = nn.Linear(in_features=out_channels, out_features=net_params['n_vars'])

    def forward(self, g, features):
        for i, layer in enumerate(self.layers):
            features = layer(g, features)
            if i == 0 and self.additional_feature:
                features = torch.cat([features, g.ndata['additional_feature']], dim=-1)
            if i != len(self.layers) - 1:  # Don't apply dropout after the last layer
                features = self.dropout(features)

        g.ndata['embedding'] = features
        if self.readout_type == 'mean':
            features = dgl.mean_nodes(g, 'embedding')
        elif self.readout_type == 'max':
            features = dgl.max_nodes(g, 'embedding')
        else:
            raise ValueError(f"Invalid readout type: {self.readout_type}")

        return self.predict(features), g


# if __name__ == "__main__":
#     import json
#     with open(r'F:\projects\AiyingT1MultimodalFusion\scripts\configs\GCN.json') as f:
#         config = json.load(f)
#     model = GCN(config['net_params'])
#     print(model)
