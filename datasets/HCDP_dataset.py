# -*- coding: utf-8 -*-
# @Time    : 9/27/2023 7:42 PM
# @Author  : Gang Qu
# @FileName: HCDP_dataset.py

import os
import torch
from torch.utils.data import Dataset
import os
import pickle
from scripts.utils.utils import group_by_attributes, get_labels_for_subjects_and_attributes, get_connection
import sys
import dgl
import numpy as np

sys.path.append("F:/projects/AiyingT1MultimodalFusion/")

# DATA_PATH = os.path.abspath('F:\projects\AiyingT1MultimodalFusion\data\hcdp\HCDPdata.pkl')
DATA_PATH = os.path.abspath('F:\projects\AiyingT1MultimodalFusion\data\hcdp\HCDPdata_int.pkl')
DATA_PATH = os.path.abspath('F:\projects\AiyingT1MultimodalFusion\data\hcdp\HCDPdata_int_coup.pkl')
DATA_PATH = os.path.abspath('F:\projects\AiyingT1MultimodalFusion\data\hcdp\HCDPdata_int_coup_schaefer.pkl')

NUM_ROI = 200
class HCDP(Dataset):
    def __init__(self, task, x_attributes, y_attribute, rawdata_path=DATA_PATH,
                 dataset_path=r"F:\projects\AiyingT1MultimodalFusion\scripts\datasets\hcdp_dataset",
                 buckets_dict={"Age in year": [(8, 12), (12, 18), (18, 23)]},
                 attributes_to_group=None, suffix=None):

        assert task in ['classification', 'regression'], "Task should be either 'classification' or 'regression'"
        dataset_path += f"_{task}" + (f"_{suffix}" if suffix else "") + ".pkl"
        self.task = task
        self.x_attributes = x_attributes
        self.y_attributes = y_attribute if isinstance(y_attribute, list) else [y_attribute]
        self.buckets_dict = buckets_dict
        self.attributes_to_group = attributes_to_group

        with open(rawdata_path, 'rb') as f:
                data = pickle.load(f)
        self.processed_data = self.process_data(data)


    def process_data(self, data):
        if self.task == 'classification':
            grouped_data = group_by_attributes(data, self.buckets_dict, self.attributes_to_group)
            labels_for_subjects, attribute_for_labels = get_labels_for_subjects_and_attributes(grouped_data, self.buckets_dict)
            labels_dict = dict(labels_for_subjects)
        else:  # regression
            labels_dict = {
                subject_id: [attributes[attr] for attr in self.y_attributes if
                             attr in attributes and attributes[attr] is not None]
                for subject_id, attributes in data.items()
                if all(attr in attributes and attributes[attr] is not None for attr in self.y_attributes)
            }

        processed = []
        # for subject_id, attributes in data.items():
        #     if all(attr in attributes and attributes[attr] is not None for attr in self.x_attributes):
        #         if self.task == 'classification':
        #             y = labels_dict.get(subject_id, None)
        #
        #             if y >= len(list(self.buckets_dict.values())[0]):
        #                 continue
        #         x = [attributes[attr] for attr in self.x_attributes]
        #         y = labels_dict.get(subject_id, None)
        #
        #         if isinstance(y, list):
        #             y = [float(value) if value and value.strip() != '' else None for value in y]
        #
        #             if any(val is None for val in y):  # skip subjects with missing y values
        #                 continue
        #         else:
        #             try:
        #                 y = float(y)
        #             except (ValueError, TypeError):
        #                 continue  # skip subjects if y can't be converted to a float
        #
        #         processed.append((subject_id, x, y))
        #     else:
        #         print(f"Skipping subject {subject_id} due to missing attributes")

        for subject_id, attributes in data.items():
            # Ensure all required x_attributes are present and not None
            if not all(attr in attributes and attributes[attr] is not None for attr in self.x_attributes):
                print(f"Skipping subject {subject_id} due to missing attributes")
                continue

            y = labels_dict.get(subject_id)  # Initial retrieval of y

            # For classification, check if y is within the valid range; skip if not
            if self.task == 'classification' and (y is None or y >= len(list(self.buckets_dict.values())[0])):
                continue

            # Preparation of x attributes
            x = [attributes[attr] for attr in self.x_attributes]

            # Handling y for tasks other than classification or when y is a list
            if isinstance(y, list):
                # Convert list elements to float or None; skip subject if any element is None
                y = [float(value) if value and value.strip() != '' else None for value in y]
                if any(val is None for val in y):
                    continue
            else:
                # Attempt to convert y to float for non-list y; skip subject on failure
                try:
                    y = None if y is None else float(y)
                except (ValueError, TypeError):
                    continue

            # Final check to ensure y is not None after all conversions
            if y is None:
                continue

            processed.append((subject_id, x, y))

        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):

        subject_id, x, y = self.processed_data[idx]
        x_tensors = [torch.tensor(lst, dtype=torch.float32) for lst in x]

        if isinstance(y, list):
            y_tensor = torch.tensor(y, dtype=torch.float32)
        elif self.task == 'classification':
            y_tensor = torch.tensor(int(y), dtype=torch.int64)
        else:  # regression
            y_tensor = torch.tensor(y, dtype=torch.float32)

        return subject_id, x_tensors, y_tensor

    def get_labels(self):
        # Return the labels of the dataset
        labels = []
        for i in range(self.__len__()):
            labels.append(self.processed_data[i][2])
        return labels


class HCDP_DGL(HCDP):
    def __init__(self, *args, mean=None, std=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        # Get data using the parent class method
        subject_id, x, y = super().__getitem__(idx)

        # Handle multiple x_attributes. If x is a tuple of matrices, use the first one for adjacency.
        x_for_adjacency = x[0].numpy()

        # Compute the adjacency matrix from the chosen matrix
        adj_matrix = get_connection(x_for_adjacency, k=30)
        threshold_value = 1e-4
        adj_matrix[np.abs(adj_matrix) < threshold_value] = 0
        # Convert the adjacency matrix to a DGL graph
        edge_indices = torch.nonzero(torch.tensor(adj_matrix), as_tuple=True)
        num_nodes = adj_matrix.shape[0]
        g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=num_nodes)

        # Attach node features to the graph

        if isinstance(x, (tuple, list)):
            # Convert each matrix in x to a tensor and concatenate them along the feature dimension
            x_tensor = torch.cat([torch.as_tensor(mat, dtype=torch.float32).clone().detach() for mat in x], dim=-1)
        elif isinstance(x, torch.Tensor) and x.dim() > 2:
            # If x is already a tensor with 3 dimensions, concatenate along the last dimension
            x_tensor = torch.cat(x.unbind(0), dim=-1)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)

        if x_tensor.shape[-1] % NUM_ROI != 0:
            g.ndata['x'] = x_tensor[:, :x_tensor.shape[-1] // NUM_ROI*NUM_ROI]
            g.ndata['additional_feature'] = x_tensor[:, x_tensor.shape[-1] // NUM_ROI * NUM_ROI:]
            if self.mean is not None and self.std is not None:
                # Apply standardization
                g.ndata['additional_feature'] = (g.ndata['additional_feature'] - self.mean) / (self.std + 1e-6)
            g.ndata['h'] = g.ndata['x']
        else:
            g.ndata['x'] = x_tensor
            g.ndata['h'] = g.ndata['x']


        # Convert y to tensor (can be a scalar or a vector depending on y_attribute length)
        if isinstance(y, list):
            y_tensor = torch.tensor(y, dtype=torch.float32)
        elif self.task == 'classification':
            y_tensor = torch.tensor(int(y), dtype=torch.int64)
        else:  # regression
            y_tensor = torch.tensor(y, dtype=torch.float32)

        return subject_id, g, y_tensor


# if __name__ == "__main__":
#     # buckets = {"Age in year": [(8, 12), (12, 18), (18, 23)]}
#     # attributes_group = ["Age in year"]
#     # dataset1 = HCDP( task='classification',
#     #                x_attributes=['FC', 'SC', 'CT'], y_attribute='age',
#     #                buckets_dict=buckets, attributes_to_group=attributes_group)
#     # print(dataset1[0], len(dataset1))
#     # for i in range(200):
#     #     print(dataset1[i])
#     attributes_group = ["nih_totalcogcomp_ageadjusted"]
#     buckets = {"nih_totalcogcomp_ageadjusted": [(60, 81), (129, 150)]}
#     dataset2 = HCDP_DGL(task='classification',
#                    x_attributes=['FC', 'SC', 'CT'], y_attribute=['nih_totalcogcomp_ageadjusted'],
#                    buckets_dict=buckets, attributes_to_group=attributes_group)
#     print(len(dataset2))
#
#     indices = [idx for idx in range(len(dataset2)) if dataset2[idx][2] != 2]


