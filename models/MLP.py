# -*- coding: utf-8 -*-
# @Time    : 10/4/2023 8:34 AM
# @Author  : Gang Qu
# @FileName: MLP.py
import dgl
import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, net_params):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        in_channels = net_params['in_channels']
        hidden_channels = net_params['hidden_channels']
        out_channels = net_params['out_channels']
        self.num_layers = net_params['num_layers']

        dropout = net_params['dropout']
        self.readout_type = net_params['readout']
        if net_params['activation'] == "elu":
            activation = torch.nn.ELU()
        else:
            activation = torch.nn.ReLU()
        self.additional_feature = net_params['additional_feature']
        if self.num_layers >= 1:
            nnlayer = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels-self.additional_feature), activation)
        else:
            nnlayer = torch.nn.Sequential(torch.nn.Linear(in_channels+self.additional_feature, net_params['n_vars']), nn.Identity())
        self.layers.append(nnlayer)
        self.dropout = nn.Dropout(dropout)

        for i in range(1, self.num_layers):
            if i != self.num_layers - 1:
                nnlayer = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), activation)
                self.layers.append(nnlayer)
            else:
                nnlayer = torch.nn.Sequential(torch.nn.Linear(hidden_channels, out_channels), activation)
                self.layers.append(nnlayer)
        if self.num_layers >= 1:
            self.predict = nn.Linear(in_features=out_channels, out_features=net_params['n_vars'])
        else:
            self.predict = nn.Identity(out_channels)

    def forward(self, g, features):
        if self.num_layers == 0 and self.additional_feature:
            features = torch.cat([features, g.ndata['additional_feature']], dim=-1)
        for i, layer in enumerate(self.layers):
            features = layer(features)
            if i == 0 and self.num_layers >= 1 and self.additional_feature:
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

#
# if __name__ == "__main__":
#     import json
#     with open(r'F:\projects\AiyingT1MultimodalFusion\scripts\configs\MLP.json') as f:
#         config = json.load(f)
#     model = MLP(config['net_params'])
#     print(model)
