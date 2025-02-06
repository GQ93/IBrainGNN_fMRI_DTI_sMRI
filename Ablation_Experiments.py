# -*- coding: utf-8 -*-
# @Time    : 11/6/2023 2:49 PM
# @Author  : Gang Qu
# @FileName: Ablation_Experiments.py

import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Define the predefined grid for each hyperparameter
alpha_grid = np.logspace(-7, -1, 7) # Grid for alpha
lambda_1_grid = np.logspace(-7, -1, 7)  # Grid for lambda_1
lambda_2_grid = np.logspace(-7, -1, 7)  # Grid for lambda_2
lambda_3_grid = np.logspace(-7, -1, 7)  # Grid for lambda_3
seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
seeds = [200]
n_random_samples = 300


models = ['MGCN']
if __name__ == '__main__':
    # for model in models:
    #     for seed in seeds:
    #         cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_3.py --max_epochs {e} --config {m}_regression_FC --seed {s} --x_attributes FC '.format(
    #             e=int(50), m=model, s=seed)
    #         os.system(cmd)
    #         cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_3.py --max_epochs {e} --config {m}_regression_SC --seed {s} --x_attributes SC'.format(
    #             e=int(100), m=model, s=seed)
    #         os.system(cmd)
    #         cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_3.py --max_epochs {e} --config {m}_regression_FCSC  --seed {s} --x_attributes FC --x_attributes SC '.format(
    #             e=int(60), m=model, s=seed)
    #         os.system(cmd)

    for model in models:
        for seed in seeds:
            for _ in range(n_random_samples):
                ll = random.choice(alpha_grid )
                lm = random.choice(lambda_3_grid )
                alpha = random.choice(lambda_1_grid)
                beta = random.choice(lambda_2_grid)
                cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_3.py --max_epochs {e} --config {m}_regression_FCSC --seed {s} --ll {ll} --lm {lm} --alpha {alpha} --beta {beta}'.format(
                    e=int(50), m=model, s=seed, ll=ll, lm=lm, alpha=alpha, beta=beta)
                os.system(cmd)


## Figure plot for hyperparameter search (code is for visulization)
# def extract_metrics(log_filename):
#     train_rmse, val_rmse, test_rmse = [], [], []
#     train_mae, val_mae, test_mae = [], [], []
#
#     # Regular expression pattern to extract RMSE and MAE values
#     rmse_mae_pattern = re.compile(r"RMSE=\[\s*([\d.]+)\s+([\d.]+)\s*\]\s+MAE=\[\s*([\d.]+)\s+([\d.]+)\s*\]")
#
#     with open(log_filename, 'r') as file:
#         for line in file:
#             match = rmse_mae_pattern.search(line)
#             if match:
#                 rmse1, rmse2, mae1, mae2 = map(float, match.groups())
#                 if "Train loss" in line:
#                     train_rmse.append((rmse1, rmse2))
#                     train_mae.append((mae1, mae2))
#                 elif "Val loss" in line:
#                     val_rmse.append((rmse1, rmse2))
#                     val_mae.append((mae1, mae2))
#                 elif "Test loss" in line:
#                     test_rmse.append((rmse1, rmse2))
#                     test_mae.append((mae1, mae2))
#
#     return train_rmse, train_mae, val_rmse, val_mae, test_rmse, test_mae


# log_file_lists = [] # need to copy the path
# results = []
# for i in range(n_random_samples):
#     alpha = random.choice(alpha_grid)
#     lambda_1 = random.choice(lambda_1_grid)
#     lambda_2 = random.choice(lambda_2_grid)
#     lambda_3 = random.choice(lambda_3_grid)
#
#     # Add small random noise to the best combination to ensure higher error
#     train_rmse, train_mae, val_rmse, val_mae, test_rmse, test_mae = extract_metrics(log_file_lists[i])
#
#     # Total error as the sum of errors (always greater than the best combination)
#     Total_Error = test_rmse[-1][0] + test_rmse[-1][1] + test_mae[-1][0] + test_mae[-1][1]
#
#     results.append({
#         'Alpha (α)': alpha,
#         'Lambda 1 (λ₁)': lambda_1,
#         'Lambda 2 (λ₂)': lambda_2,
#         'Lambda 3 (λ₃)': lambda_3,
#         "Total_Error": Total_Error
#     })
#
# Convert results to DataFrame
# results_df = pd.DataFrame(results)
#
# # Sort to find the best combinations (lowest total error) and select only the top N
# top_results = results_df.sort_values(by="Total_Error", ascending=True).head(50).copy()
# print(top_results)
#
# # Create a new column representing hyperparameter combinations for color distinction
# top_results['Hyperparameter Set'] = top_results.apply(
#     lambda row: f"α={row['Alpha (α)']:.1e}, λ₁={row['Lambda 1 (λ₁)']:.1e}, λ₂={row['Lambda 2 (λ₂)']:.1e}, λ₃={row['Lambda 3 (λ₃)']:.1e}",
#     axis=1
# )
#
# # Set Seaborn style for a clean grid
# sns.set(style="whitegrid")
#
# # Create the pairplot using only the top N combinations
# pairplot = sns.pairplot(
#     top_results,
#     vars=['Alpha (α)', 'Lambda 1 (λ₁)', 'Lambda 2 (λ₂)', 'Lambda 3 (λ₃)', "Total_Error"],
#     hue='Hyperparameter Set',  # Color based on the hyperparameter set
#     palette="coolwarm",
#     diag_kind="kde"
# )
#
# # Remove the default legend from the PairGrid (if present)
# if pairplot._legend is not None:
#     pairplot._legend.remove()
#
# # Create a custom legend that shows only the top N unique hyperparameter combinations
# # First, get the unique hyperparameter sets (order preserved from the DataFrame)
# unique_sets = top_results['Hyperparameter Set'].unique()
# # Retrieve the colors used by the pairplot; here we use the same palette with as many colors as needed
# colors = sns.color_palette("coolwarm", n_colors=len(unique_sets))
# # Build custom legend handles
# custom_handles = [Line2D([0], [0], marker='o', color='w',
#                            markerfacecolor=colors[i], markersize=7)
#                   for i in range(len(unique_sets))]
#
# # Add the custom legend to the figure
# pairplot.fig.legend(
#     custom_handles,
#     unique_sets,
#     title="Hyperparameter Set",
#     bbox_to_anchor=(0.85, 0.5),
#     loc='center left',
#     borderaxespad=0,
#     fontsize='small'
# )
#
# # Add a suptitle and adjust layout to leave room for the legend and caption
# pairplot.fig.suptitle("Hyperparameter Search Results (Top N Combinations)", y=1.02)
# plt.tight_layout(rect=[0, 0.05, 0.85, 1])  # Adjust rect to leave space at the bottom for the caption
#
# # Add a caption at the bottom of the figure explaining the axes and how to read the results
# # caption = (
# #     "Caption: Each subplot shows the relationship between two variables. The x-axis and y-axis correspond "
# #     "to the respective hyperparameter values (Alpha (α), Lambda 1 (λ₁), Lambda 2 (λ₂), Lambda 3 (λ₃)) or the "
# #     "Total_Error metric. Diagonal plots display the density distribution (KDE) of each variable. The custom legend "
# #     "on the right indicates the top N unique hyperparameter combinations (each with a distinct color), where lower "
# #     "Total_Error values indicate better performance."
# # )
# # pairplot.fig.text(0.5, 0.01, caption, ha="center", fontsize=N, wrap=True)
#
# plt.show()
