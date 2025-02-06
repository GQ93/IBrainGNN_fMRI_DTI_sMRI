# -*- coding: utf-8 -*-
# @Time    : 8/16/2022 10:42 AM
# @Author  : Gang Qu
# @FileName: utils.py
import os
import torch
import numpy as np
import copy
import logging.config
import random
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse as sp
import dgl
import torch.nn.functional as F
from scripts.models.model_loss import weighted_mse_loss, classification_loss
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_curve, auc, f1_score, confusion_matrix, recall_score, precision_score
LOGGING_DIC = {
    'version': 1.0,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format':
                '%(asctime)s %(threadName)s:%(thread)d [%(name)s] %(levelname)s [%(pathname)s:%(lineno)d] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'simple': {
            'format': '%(asctime)s [%(name)s] %(levelname)s %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'test': {
            'format': '%(asctime)s %(message)s',
        },
    },
    'filters': {},
    'handlers': {
        'console_debug_handler': {
            'level': 'DEBUG',  # 日志处理的级别限制
            'class': 'logging.StreamHandler',  # 输出到终端
            'formatter': 'simple'  # 日志格式
        },
        'file_info_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件,日志轮转
            'filename': 'user.log',
            'maxBytes': 1024 * 1024 * 10,  # 日志大小 10M
            'backupCount': 10,  # 日志文件保存数量限制
            'encoding': 'utf-8',
            'formatter': 'standard',
        },
        'file_debug_handler': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',  # 保存到文件
            'filename': 'test.log',  # 日志存放的路径
            'encoding': 'utf-8',  # 日志文件的编码
            'formatter': 'test',
        },
    },
    'loggers': {
        'logger1': {  # 导入时logging.getLogger时使用的app_name
            'handlers': ['console_debug_handler'],  # 日志分配到哪个handlers中
            'level': 'DEBUG',
            'propagate': False,
        },
        'logger2': {
            'handlers': ['console_debug_handler', 'file_debug_handler'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}



def compute_batch_laplacian(gs):
    """
    Compute the Laplacian matrix for each graph in a batch of graphs.

    Args:
        gs (dgl.DGLGraph): A batched DGLGraph.

    Returns:
        torch.Tensor: A tensor of Laplacian matrices for each graph in the batch,
                      shape (Batch, N_nodes, N_nodes).
    """
    device = gs.device
    laplacians = []

    # Unbatch the graph to process each graph independently
    for g in dgl.unbatch(gs):
        # Get the adjacency matrix for a single graph, convert to dense and ensure it's on the right device
        A = g.adjacency_matrix().to_dense().to(device)

        # Get degrees for each node and construct the diagonal degree matrix for a single graph
        degrees = g.in_degrees().float().to(device)
        D = torch.diag(degrees).to(device)

        # Compute the Laplacian for the single graph: L = D - A
        L = D - A
        laplacians.append(L)

    # Optionally, you can batch them back if needed
    return torch.stack(laplacians)
    # return laplacians  # Return as a list if you prefer to handle them separately


def get_logger(name, path='results/loggers'):
    """
    get the logger
    :param name: name of logger file
    :type name:
    :return:
    :rtype:
    """
    log_config = copy.deepcopy(LOGGING_DIC)
    if not os.path.exists(path):
        os.makedirs(path)
    log_config['handlers']['file_debug_handler']['filename'] = os.path.join(path, name)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger('logger2')
    return logger


def seed_it(seed):
    """
    set random seed for reproducibility
    :param seed: seed number
    :type seed: int
    :return:
    :rtype:
    """
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


# def log_metrics(epoch_loss, predicted_scores, target_scores, task_type, phase, epoch, writer, logger):
#     if task_type == "regression":
#         rmse = evaluate_mat(predicted_scores, target_scores, method='RMSE')
#         mae = evaluate_mat(predicted_scores, target_scores, method='MAE')
#         log_or_print('{} loss={} RMSE={} MAE={}'.format(phase, epoch_loss, rmse, mae), logger)
#         if writer:
#             writer.add_scalar('Loss/' + phase, epoch_loss, epoch)
#             for idx, (r, m) in enumerate(zip(rmse, mae)):
#                 writer.add_scalar('Metrics/RMSE_{}_{}'.format(idx, phase), r, epoch)
#                 writer.add_scalar('Metrics/MAE_{}_{}'.format(idx, phase), m, epoch)
#     elif task_type == "classification":
#         # Convert predicted scores to class label
#         predicted_scores = np.concatenate(predicted_scores, axis=0)
#
#         # Convert predicted scores to class label
#         predicted_labels = np.argmax(predicted_scores, axis=1)
#         true_labels = np.concatenate(target_scores).ravel()
#
#         correct_predictions = np.sum(predicted_labels == true_labels)
#
#         # Calculate the accuracy
#         accuracy = correct_predictions / len(true_labels) * 100
#
#         # Assuming there are n_classes
#         n_classes = predicted_scores.shape[1]
#
#         # Compute ROC and AUC for either binary or multiclass classification
#         if n_classes == 2:
#             fpr, tpr, _ = roc_curve(true_labels, predicted_scores[:, 1])
#             roc_auc = auc(fpr, tpr)
#             # For Tensorboard: log the curve points
#             if writer:
#                 writer.add_scalar('Metrics/AUC_' + phase, roc_auc, epoch)
#                 for i in range(len(fpr)):
#                     writer.add_scalars('ROC_curve_' + phase,
#                                        {'False Positive Rate': fpr[i], 'True Positive Rate': tpr[i]}, i)
#         else:
#             # Binarize the labels for one-vs-all ROC computation
#             true_labels_bin = label_binarize(true_labels, classes=np.arange(n_classes))
#             false_positive_rate = dict()
#             true_positive_rate = dict()
#             roc_auc = dict()
#             for i in range(n_classes):
#                 false_positive_rate[i], true_positive_rate[i], _ = roc_curve(true_labels_bin[:, i],
#                                                                              predicted_scores[:, i])
#                 roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])
#
#                 # For Tensorboard: log the curve points for each class
#                 if writer:
#                     writer.add_scalar('Metrics/AUC_class_' + str(i) + "_" + phase, roc_auc[i], epoch)
#                     for j in range(len(false_positive_rate[i])):
#                         writer.add_scalars('ROC_curve_class_' + str(i) + "_" + phase,
#                                            {'False Positive Rate': false_positive_rate[i][j],
#                                             'True Positive Rate': true_positive_rate[i][j]}, j)
#
#         # log the AUC values
#         if n_classes == 2:
#             log_or_print(f"{phase} Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, FPR: {fpr:.4f}, TPR: {tpr:.4f}, AUC={roc_auc:.4f}", logger)
#
#         else:
#             for i in range(n_classes):
#                 log_or_print(f"{phase} Loss: {epoch_loss:.4f}, AUC for class {i} = {roc_auc[i]:.4f}, Accuracy: {accuracy:.2f}%",
#                              logger)



def log_metrics(epoch_loss, predicted_scores, target_scores, task_type, phase, epoch, writer, logger):
    if task_type == "regression":
        rmse = evaluate_mat(predicted_scores, target_scores, method='RMSE')
        mae = evaluate_mat(predicted_scores, target_scores, method='MAE')
        log_or_print(f'{phase} loss={epoch_loss} RMSE={rmse} MAE={mae}', logger)
        if writer:
            writer.add_scalar('Loss/' + phase, epoch_loss, epoch)
            for idx, (r, m) in enumerate(zip(rmse, mae)):
                writer.add_scalar(f'Metrics/RMSE_{idx}_{phase}', r, epoch)
                writer.add_scalar(f'Metrics/MAE_{idx}_{phase}', m, epoch)

    elif task_type == "classification":

        # Flatten predicted scores and true labels

        predicted_scores = np.concatenate(predicted_scores, axis=0)

        predicted_labels = np.argmax(predicted_scores, axis=1)

        true_labels = np.concatenate(target_scores).ravel()

        # Calculate accuracy

        correct_predictions = np.sum(predicted_labels == true_labels)

        accuracy = correct_predictions / len(true_labels) * 100

        # Handling different confusion matrix sizes

        cm = confusion_matrix(true_labels, predicted_labels)

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], 0, 0, 0) if cm.size == 1 else np.pad(cm.ravel(), (
        0, 4 - cm.size))

        # Calculate specificity, sensitivity, and F1 score

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivity = recall_score(true_labels, predicted_labels, zero_division=0)

        f1 = f1_score(true_labels, predicted_labels, zero_division=0)

        # Calculate AUC for binary classification

        if predicted_scores.shape[1] == 2:

            fpr, tpr, _ = roc_curve(true_labels, predicted_scores[:, 1])

            roc_auc = auc(fpr, tpr)

            if writer:

                writer.add_scalar('Metrics/AUC_' + phase, roc_auc, epoch)

                writer.add_scalar('Metrics/Sensitivity_' + phase, sensitivity, epoch)

                writer.add_scalar('Metrics/Specificity_' + phase, specificity, epoch)

                writer.add_scalar('Metrics/F1_Score_' + phase, f1, epoch)

                for i in range(len(fpr)):
                    writer.add_scalars(f'ROC_curve_{phase}',
                                       {'False Positive Rate': fpr[i], 'True Positive Rate': tpr[i]}, i)

            log_or_print(
                f"{phase} Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}",
                logger)


def evaluate_mat(predicted, target, method):
    """
    Evaluate the RMSE or MAE between predicted and target data.

    Parameters:
    - predicted (Tensor or list of Tensors): The predicted scores.
    - target (Tensor or list of Tensors): The target scores.
    - method (str): The method to use for evaluation ('RMSE' or 'MAE').

    Returns:
    numpy.ndarray: The evaluation result.
    """

    def to_single_tensor(data):
        """
        Convert data to a single PyTorch tensor.

        Parameters:
        - data: List of PyTorch tensors/NumPy ndarrays or a single tensor/ndarray

        Returns:
        - Tensor: A single concatenated PyTorch tensor
        """
        if isinstance(data, list):
            # Ensure all elements are torch tensors before concatenation
            data = [torch.tensor(d).to(dtype=torch.float32, device='cuda') if isinstance(d, np.ndarray) else d.to(
                dtype=torch.float32, device='cuda') for d in data]
            data = torch.cat(data, dim=0)
        elif isinstance(data, np.ndarray):
            data = torch.tensor(data).to(dtype=torch.float32, device='cuda')
        # Ensure the resulting tensor is on the right device and dtype
        return data.to(dtype=torch.float32, device='cuda')

    # Convert lists of tensors to single tensors if needed
    predicted = to_single_tensor(predicted)
    target = to_single_tensor(target)

    # Calculate the difference between the predicted and target scores.
    res = predicted - target

    if method == 'RMSE':
        res = res ** 2
        # Assuming res is a PyTorch tensor for efficient computation
        res = torch.sqrt(torch.sum(res, dim=0) / res.shape[0]).cpu().numpy()

    elif method == 'MAE':
        # Assuming res is a PyTorch tensor for efficient computation
        res = (torch.sum(torch.abs(res), dim=0) / res.shape[0]).cpu().numpy()

    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'RMSE' or 'MAE'.")

    return res


def log_or_print(message, logger):
    if logger:
        logger.info(message)
    else:
        print(message)


class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            self.min_delta = validation_loss - train_loss
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0


def get_bucket(value, buckets):
    """Return the bucket/range for a given value."""
    for start, end in buckets:
        if start <= value < end:
            return f"{start}-{end}"
    return "Other"


def safe_float_conversion(value):
    try:
        return float(value)
    except ValueError:
        return value


def group_by_attributes(data, buckets_dict, attributes_to_group):
    """
    Group subjects based on specified buckets and attributes.

    Parameters:
    - data (dict): Dictionary containing subject IDs as keys and another dictionary of attributes as values.
    - buckets_dict (dict): Dictionary specifying buckets for each attribute.
    - attributes_to_group (list): List of attributes to group by.

    Returns:
    - dict: A dictionary where keys are attribute values or combinations, and values are lists of subject IDs.
    """

    def safe_float_conversion(value):
        try:
            return float(value)
        except ValueError:
            return value

    grouped = {}

    for subject, attributes in data.items():
        keys = []
        skip_subject = False
        for attr in sorted(attributes_to_group):  # sort the attributes_to_group for consistent order
            # Skip this subject if it's missing one of the attributes_to_group
            if attr not in attributes or attributes[attr] is None:
                skip_subject = True
                break

            value = safe_float_conversion(attributes[attr])

            if attr in buckets_dict:
                keys.append(get_bucket(value, sorted(buckets_dict[attr])))  # ensure the buckets are sorted
            else:
                keys.append(value)

        if skip_subject:
            continue

        key = tuple(keys) if len(keys) > 1 else keys[0]

        if key not in grouped:
            grouped[key] = []
        grouped[key].append(subject)

    return grouped


def get_labels_for_subjects_and_attributes(grouped_data, buckets_dict):
    labels_for_subjects = []
    attribute_for_labels = {}
    current_label = 0

    # Go over each attribute in buckets_dict in the defined order
    for attribute, bucket_ranges in buckets_dict.items():
        for bucket_range in bucket_ranges:
            group = f"{bucket_range[0]}-{bucket_range[1]}"
            if group in grouped_data:
                attribute_for_labels[current_label] = group
                subjects = grouped_data[group]
                for subject in subjects:
                    labels_for_subjects.append((subject, current_label))
                current_label += 1

    # Now handle any additional groups in grouped_data that aren't in buckets_dict
    for group, subjects in grouped_data.items():
        if group not in attribute_for_labels.values():  # To avoid re-processing
            attribute_for_labels[current_label] = group
            for subject in subjects:
                labels_for_subjects.append((subject, current_label))
            current_label += 1

    return labels_for_subjects, attribute_for_labels


def get_connection(x, k=None, threshold=None, binarize=False, is_symmetric=True):
    """
    Process the matrix x to get an adjacency matrix.
    """

    def keep_k_largest_for_rows(mat, k):
        # We sort values in ascending order and then keep only the largest k for each row.
        # The rest are set to 0.
        sorted_indices = np.argsort(mat, axis=1)
        row_indices = np.arange(mat.shape[0])[:, None]
        mat[row_indices, sorted_indices[:, :-k]] = 0
        return mat

    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        x = x.cpu().numpy()

    def process_matrix(matrix):
        if k:
            matrix = keep_k_largest_for_rows(matrix, k)

        if threshold:
            matrix[matrix < threshold] = 0

        if binarize:
            matrix = np.where(matrix > 0, 1, 0)

        if is_symmetric:
            matrix = np.maximum(matrix, matrix.T)

        return matrix

    if len(x.shape) == 3:
        for idx in range(x.shape[0]):
            x[idx] = process_matrix(x[idx])
    else:
        x = process_matrix(x)

    if is_tensor:
        x = torch.from_numpy(x)

    return x


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    #  [(subid, graph, label), (subid, graph, label), ...].
    subs, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    try:
        labels_tensor = torch.stack(labels, dim=0)
    except RuntimeError as e:
        print(f"Error with labels: {labels}")
        raise e
    return subs, batched_graph, labels_tensor
