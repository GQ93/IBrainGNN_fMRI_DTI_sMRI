# -*- coding: utf-8 -*-
# @Time    : 3/11/2024 11:25 PM
# @Author  : Gang Qu
# @FileName: Classification_MASKGCN_INTER_Experiments.py
import os

seeds = [100]
models = ['MASKGCN']
if __name__ == '__main__':
    for model in models:
        for seed in seeds:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_5.py --max_epochs {e} --config {m}_classification_IQ --seed {s}'.format(e=int(70), m=model, s=seed)
            os.system(cmd)