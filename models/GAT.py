# -*- coding: utf-8 -*-
# @Time    : 4/6/2023 8:43 PM
# @Author  : Gang Qu
# @FileName: GAT.py

import dgl
import torch
import torch.nn as nn
from dgl.nn import GATConv
from torch.nn import functional as F

class GAT(nn.Module):
    def __init__(self, net_params):
        super(GAT, self).__init__()
        self.layers = torch.nn.ModuleList()

        in_channels = net_params['in_channels']
        hidden_channels = net_params['hidden_channels']
        out_channels = net_params['out_channels']
        heads = net_params['heads']
        feat_drop = net_params['feat_drop']
        attn_drop = net_params['attn_drop']
        num_layers = net_params['num_layers']
        self.readout_type = net_params['readout']
        self.additional_feature = net_params['additional_feature']
        if net_params['activation'] == "elu":
            activation = F.elu
        else:
            activation = F.relu

        self.layers.append(
            GATConv(
                in_channels, hidden_channels-self.additional_feature, num_heads=1, activation=activation,
                feat_drop=feat_drop, attn_drop=attn_drop, allow_zero_in_degree=True)
        )

        if num_layers == 2:
            self.layers.append(
                GATConv(
                    hidden_channels, out_channels, num_heads=heads, activation=activation,
                    feat_drop=feat_drop, attn_drop=attn_drop, allow_zero_in_degree=True)
            )
        elif num_layers > 2:
            for i in range(2, num_layers):
                if i != num_layers - 1:
                    self.layers.append(
                        GATConv(
                            hidden_channels * heads, hidden_channels, num_heads=heads, activation=activation,
                            feat_drop=feat_drop, attn_drop=attn_drop, allow_zero_in_degree=True)
                    )

                else:
                    self.layers.append(
                        GATConv(
                            hidden_channels * heads, out_channels, num_heads=heads, activation=activation,
                            feat_drop=0, attn_drop=0, allow_zero_in_degree=True)
                    )
        self.predict = nn.Linear(in_features=out_channels*heads, out_features=net_params['n_vars'])

    def forward(self, g, features):
        for i, layer in enumerate(self.layers):
            features = layer(g, features)
            if i == 0 and self.additional_feature:
                features = torch.mean(features, dim=1)
                features = torch.cat([features, g.ndata['additional_feature']], dim=-1)
            else:
                features = features.view(features.shape[0], -1)

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
#     with open(r'F:\projects\AiyingT1MultimodalFusion\scripts\configs\GAT.json') as f:
#         config = json.load(f)
#     model = GAT(config['net_params'])
#     print(model)
