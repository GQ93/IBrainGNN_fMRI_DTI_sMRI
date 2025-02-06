# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 8:57 PM
# @Author  : Gang Qu
# @FileName: load_model.py
from scripts.models.GCN import GCN
from scripts.models.GIN import GIN
from scripts.models.GAT import GAT
from scripts.models.MLP import MLP
from scripts.models.MASKGCN import MASKGCN
def load_model(config):
    if config['model'] == 'GCN':
        model = GCN(config['net_params'])
    elif config['model'] == 'GIN':
        model = GIN(config['net_params'])
    elif config['model'] == 'GAT':
        model = GAT(config['net_params'])
    elif config['model'] == 'MLP':
        model = MLP(config['net_params'])
    elif config['model'] == 'Linear':
        model = MLP(config['net_params'])
    elif config['model'] == 'MASKGCN':
        model = MASKGCN(config['net_params'])
    return model
