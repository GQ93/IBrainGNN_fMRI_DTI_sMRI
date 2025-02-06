# -*- coding: utf-8 -*-
# @Time    : 11/3/2023 8:25 PM
# @Author  : Gang Qu
# @FileName: Regression_GNN_Experiments.py
import os

seeds = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900]
models = ['GAT', 'GCN', 'GIN', 'MLP', 'Linear']
LR = [1e-3, 1e-3, 1e-3, 1e-3, 1e-2]
models = ['Linear']
if __name__ == '__main__':
    for model in models:
        for seed in seeds:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_1.py --max_epochs {e} --config {m}_regression --seed {s}'.format(e=int(100), m=model, s=seed)
            os.system(cmd)
