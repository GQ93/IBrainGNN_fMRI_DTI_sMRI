# -*- coding: utf-8 -*-
# @Time    : 4/6/2023 8:43 PM
# @Author  : Gang Qu
# @FileName: GCN.py

import dgl
import torch
import torch.nn as nn
from dgl.nn import GraphConv
from torch.nn import functional as F
import numpy as np
# class SymmetricMaskedGraphConv(GraphConv):
#     def __init__(self, in_feats, out_feats, num_nodes, *args, **kwargs):
#         # Initialize the GraphConv class without the num_nodes keyword
#         super(SymmetricMaskedGraphConv, self).__init__(in_feats, out_feats, *args, **kwargs)
#
#         # Now handle the num_nodes separately
#         self.num_nodes = num_nodes
#         self.raw_edge_weight = nn.Parameter(torch.ones(num_nodes, num_nodes))
#
#     @property
#     def edge_weight(self):
#         # Enforce symmetry by averaging with the transpose
#         return (self.raw_edge_weight + self.raw_edge_weight.t()) / 2
#
#     def forward(self, graph, feat):
#         with graph.local_scope():
#             # Retrieve the indices of the source and destination nodes for each edge
#             src, dst = graph.edges()
#
#             # Use the indices to look up the corresponding weights from the symmetric matrix
#             edge_weights = self.edge_weight[src, dst]
#
#             # Apply edge weights
#             graph.edata['w'] = edge_weights
#
#             # Standard message passing
#             graph.update_all(dgl.function.u_mul_e('h', 'w', 'm'),
#                              dgl.function.sum('m', 'h'))
#             rst = feat @ self.weight + graph.ndata['h']
#
#             # Apply activation if any
#             if self._activation is not None:
#                 rst = self._activation(rst)
#
#             return rst



class SymmetricMaskedGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_nodes_per_graph, shared_mask=None, *args, **kwargs):
        super(SymmetricMaskedGraphConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_nodes_per_graph = num_nodes_per_graph
        # Initialize weights for the GCN layer
        self.weight = nn.Parameter(torch.randn(in_feats, out_feats))
        # Initialize a symmetric mask matrix as a parameter for a single graph
        if shared_mask is not None:
            self.raw_edge_weight = shared_mask
        else:
            self.raw_edge_weight = nn.Parameter(torch.ones(num_nodes_per_graph, num_nodes_per_graph))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.raw_edge_weight.data.uniform_(-stdv, stdv)

    @property
    def edge_weight(self):
        # Enforce symmetry by averaging with the transpose
        return (self.raw_edge_weight + self.raw_edge_weight.t()) / 2

    def forward(self, graph, feat):
        # Determine the number of graphs in the batch
        total_num_nodes = graph.number_of_nodes()
        num_graphs_in_batch = total_num_nodes // self.num_nodes_per_graph

        # Create a block-diagonal mask for the batch
        identity = torch.eye(self.edge_weight.size(0)).to(self.edge_weight.device)
        mask = self.edge_weight + identity
        batch_mask = torch.block_diag(*[mask] * num_graphs_in_batch).to(feat.device)

        # Get the adjacency matrix of the batched graph
        adj_matrix = torch.tensor(graph.adjacency_matrix(scipy_fmt="coo").todense(), dtype=torch.float32).to(feat.device)

        # Apply the batch-compatible mask
        adj_matrix_masked = torch.sigmoid(adj_matrix) * batch_mask

        # Normalize adjacency matrix with the mask (optional normalization step)
        # Example normalization omitted for brevity

        # Perform the GCN operation (AWX)
        support = torch.matmul(feat, self.weight)
        output = torch.matmul(adj_matrix_masked, support)

        # Apply activation function if specified
        if hasattr(self, '_activation') and self._activation is not None:
            output = self._activation(output)

        return output

class MASKGCN(nn.Module):
    def __init__(self, net_params):
        super(MASKGCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        in_channels = net_params['in_channels']
        hidden_channels = net_params['hidden_channels']
        out_channels = net_params['out_channels']
        num_layers = net_params['num_layers']
        dropout = net_params['dropout']
        num_nodes = net_params['num_nodes']
        self.identity_layer = torch.nn.Identity()
        self.additional_feature = net_params['additional_feature']
        self.readout_type = net_params['readout']
        if net_params['activation'] == "elu":
            activation = F.elu
        else:
            activation = F.relu
        self.layer0 = SymmetricMaskedGraphConv(in_channels, hidden_channels-self.additional_feature, shared_mask=None,
                                                    activation=activation, allow_zero_in_degree=True, num_nodes_per_graph=num_nodes)
        self.layers.append(self.layer0)
        self.layers.append(self.identity_layer)
        self.dropout = nn.Dropout(dropout)
        for i in range(1, num_layers):
            if i != num_layers - 1:
                self.layers.append(SymmetricMaskedGraphConv(hidden_channels, hidden_channels, shared_mask=self.layer0.raw_edge_weight,
                                                            activation=activation, allow_zero_in_degree=True,
                                                            num_nodes_per_graph=num_nodes))
            else:
                self.layers.append(SymmetricMaskedGraphConv(hidden_channels, out_channels, shared_mask=self.layer0.raw_edge_weight,
                                                            activation=activation, allow_zero_in_degree=True,
                                                            num_nodes_per_graph=num_nodes))
        self.predict = nn.Linear(in_features=out_channels, out_features=net_params['n_vars'])

    def forward(self, g, features):
        for i, layer in enumerate(self.layers):
            features = layer(g, features) if not isinstance(layer, nn.Identity) else layer(features)
            if i == 0 and self.additional_feature:
                if self.additional_feature and 'additional_feature' in g.ndata:
                    additional_features = g.ndata['additional_feature']
                    if not additional_features.requires_grad:
                        # This is typically not recommended for input features unless for specific analysis
                        additional_features = additional_features.clone().detach().requires_grad_(True)
                    # Concatenation operation follows

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
#     from gradcam import GNN_GradCAM
#     with open(r'/scripts/configs/MASKGCN_regression.json') as f:
#         config = json.load(f)
#     model = MASKGCN(config['net_params'])
#     print(model)
#     grad_cam_util = GNN_GradCAM(model)
#     print(grad_cam_util )
#     # activations, gradients = grad_cam_util.generate_grad_cam(g, features, target_layer_index=0, task_type='regression')
