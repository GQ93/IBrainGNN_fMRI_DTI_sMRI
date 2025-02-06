# -*- coding: utf-8 -*-
# @Time    : 11/3/2023 8:25 PM
# @Author  : Gang Qu
# @FileName: Regression_MGNN_Experiments.py
import os

seeds = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900]
models = ['MASKGCN']
if __name__ == '__main__':
    for model in models:
        for seed in seeds:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_3.py --max_epochs {e} --config {m}_regression --seed {s}'.format(e=int(100), m=model, s=seed)
            os.system(cmd)
