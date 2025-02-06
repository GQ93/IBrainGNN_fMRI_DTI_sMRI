# -*- coding: utf-8 -*-
# @Time    : 3/3/2024 8:25 PM
# @Author  : Gang Qu
# @FileName: Regression_MASKGCN_INTER_Experiments.py
import os

seeds = [100]
models = ['MASKGCN']
if __name__ == '__main__':
    for model in models:
        for seed in seeds:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_4.py --max_epochs {e} --config {m}_regression --seed {s}'.format(e=int(150), m=model, s=seed)
            os.system(cmd)