# IBrainGNN_fMRI_DTI_sMRI

This framework implements graph neural networks for predicting cognitive scores using neuroimaging data. The system
supports both regression and classification tasks with multiple experimental configurations.

---

## Project Structure
```commandline
IBrainGNN_fMRI_DTI_sMRI
├── configs/                         # Model configuration files
├── datasets/                        # Custom dataset loader
├── models/                          # Model implementations
├── results/                         # Output directory
├── train/                           # Training/testing utilities and helpers
├── utlis/                           # Helper functions
├── main_x.py                        # Primary entry point
├── *_Experiments.py                 # Scripts for running batch experiments, configure as necessary
├── env_requirements.txt             # List of Python dependencies required for the project
└── BriannetVisual.R                 # Visualization
```
---

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Data](#data)
4. [Usage](#usage)
    - [General Arguments](#general-arguments)
    - [Running `main_x.py`](#running-main_xpy)
5. [Results Management](#Results-Management)
6. [TensorBoard Visualization](#tensorboard-visualization)
7. [Citation](#citation)

---
## Overview

- Multi-modal integration of Functional Connectivity (fMRI), Structural Connectivity (DTI), and Cortical Thickness (
  sMRI)  - Support for multiple GNN backbones (e.g., GCN, GAT, GIN, and MLP baselines).
- Flexible configuration through JSON files (dropout, learning rate, architecture details).
- Automatic splitting into train, validation, and test sets based on user-defined ratios.
- Logging with `tensorboard` and checkpoint saving for reproducibility.

The main Python scripts (`main_1.py`, `main_2.py`, etc.) each focus on a different experimental setup or use slightly
different hyperparameters and data processing strategies. However, the overarching training & validation flow remains
consistent across them.

---

## Environment Setup

1. **Clone the Repository**
```bash
git clone https://github.com/GQ93/IBrainGNN_fMRI_DTI_sMRI.git
cd IBrainGNN_fMRI_DTI_sMRI
```

2. **Create (and Activate) a Virtual Environment**

    We recommend using Conda or venv, though please note that our testing has been performed only on Windows.
```bash
conda create -n IBrainGNN_fMRI_DTI_sMRI python=3.9 -y
conda activate IBrainGNN_fMRI_DTI_sMRI
```
3. **Install Dependencies**
   
    Ensure you have PyTorch, DGL, and other packages installed.
```bash
pip install -r env_requirements.txt
```
---
## Data

* These experiments rely on a brain connectivity dataset (e.g., HCDP dataset), which is part of the Adolescent Brain
  Cognitive Development (ABCD) study https://abcdstudy.org/.
* The dataset is expected to be processed into DGL graphs (see datasets/HCDP_dataset.py for details).
* Modify --dataset and path references in the code or config to point to your local data (if needed).
---
## Usage

### General Arguments

Most main_X.py scripts share a common argument set. For example:

|     Argument     |                               Description                                |                               Default                                |
|:----------------:|:------------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|       --1r       |                              Learning rate.                              |                                 1e-2                                 |
|   --batch_size   |                        Batch size for DataLoader.                        |                                  4                                   |
|   --max_epochs   |                    Maximum number of training epochs.                    |                                  1                                   |
|       --L2       |                    L2 (weight decay) regularization.                     |                                1 e-6                                 |
|    --dropout     |                       Dropout rate (0 to disable).                       |                                  0                                   |
|      --seed      |                     Random seed for reproducibility.                     |                                 100                                  |
|     --config     |     Name of the JSON configuration file in configs/ (without .json).     |                            GIN_regression                            |
|      --task      |                 Task type: regression or classification.                 |                              regression                              |
|  --x_attributes  | Input attribute(s) to include (e.g., x_attributes FC --x_attributes SC). |                          ['FC', 'SC', 'CT']                          |
|  --y_attribute   |                    Target attribute(s) for training.                     | ```['nih_crycogcomp_ageadjusted', 'nih_fluidcogcomp_ageadjusted']``` |
| attributes_group |                   Grouping attributes (buckets, etc.).                   |                           ["Age in year"]                            |
| --train_val_test |                  Train, validation, test split ratios.                   |                            [0.7,0.1,0.2]                             |
|    --dataset     |                         Dataset name identifier.                         |                                dglHCP                                |
|     --sparse     |   Sparsity parameter for constructing kNN or similar graph structure.    |                                  30                                  |
|  --weight_score  |              Weight for each target in multi-target tasks.               |                              [0.5,0.5]                               |

---

## Running main_x.py

1. **Train**
   
   The script will automatically train on the training split, evaluate on validation and test, and save the best model
   checkpoint, for example:
```bash
python main_1.py \
--lr 0.001 \
--batch_size 4 \
--max_epochs 100 \
--seed 123 \
--config GIN_regression \
--task regression \
--x_attributes FC \
--x_attributes SC \
--y_attribute nih_crycogcomp_ageadjusted \
--y_attribute nih_fluidcogcomp_ageadjusted \
--train_val_test 0.7 0.1 0.2 \
--dataset dglHCP \
--sparse 30 \
--weight_score 0.5 0.5
```
2. **Evaluation**
   
   This script also evaluates during training (on both validation and test sets). If you want to test a saved checkpoint
   separately, simply ensure you load that checkpoint (e.g., modify the code or set config['pretrain'] and
   pretrain_model_name in your config). You can look at the logged results in TensorBoard or in the console output to
   assess performance.
---

## Results Management

* **Logs** are saved under:
    * results/loggers/ for per-epoch console logs.
    * results/runs/ for TensorBoard logs.
    * **Checkpoints** are automatically saved to results/<TIMESTAMP>_<DATASET>_<...>.pth.
      A typical saved checkpoint dictionary includes:
```bash
{
 'model': model.state_dict(),
 'optimizer': optimizer.state_dict()
}
```
* Outputs are organized as:
```
results/
  ├── hcdp/pretrained/               # Model checkpoints
  ├── loggers/                       # Training logs
  └── runs/                          # TensorBoard records
```
---

## TensorBoard Visualization

The code uses **TensorBoard** via SummaryWriter to track training metrics such as loss curves over training epochs. You
can visualize them with:
```bash
    tensorboard --logdir results/runs
```
Then navigate to the reported local server address in your web browser (e.g., http://localhost:6006).

---

## Citation
If you find this code useful in your work or research, please consider citing the corresponding publication (if applicable). 
```commandline
@article{QU2025103570,
title = {Integrated brain connectivity analysis with fMRI, DTI, and sMRI powered by interpretable graph neural networks},
journal = {Medical Image Analysis},
volume = {103},
pages = {103570},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2025.103570},
url = {https://www.sciencedirect.com/science/article/pii/S1361841525001173},
author = {Gang Qu and Ziyu Zhou and Vince D. Calhoun and Aiying Zhang and Yu-Ping Wang}
}
```

The human connectome project-development (HCP-D) data are publicly available ( https://www.humanconnectome.org/study/hcp-lifespan-development). The data supporting the findings of this study are available through the National Institute of Mental Health (NIMH) Data Archive at DOI: 10.15154/qcw2-dq85

---

License

This project is licensed under the MIT License. See the LICENSE file for details.

Author: Gang Qu

Feel free to open an issue or pull request for any improvements or bug fixes!