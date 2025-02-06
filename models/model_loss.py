# -*- coding: utf-8 -*-
# @Time    : 10/3/2023 9:06 PM
# @Author  : Gang Qu
# @FileName: model_loss.py
import torch.nn.functional as F
import torch


def weighted_mse_loss(output, target, weights=None):
    """
    Weighted Mean Squared Error (MSE) for multivariate regression task.
    Args:
        output: Predicted output from the model.
        target: Ground truth.
        weights: Optional tensor of weights for each variable.
    Returns:
        Weighted loss value.
    """
    if weights is None:
        return F.mse_loss(output, target)

    squared_diffs = (output - target) ** 2

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=squared_diffs.device).view(1, -1)
    weighted_squared_diffs = weights_tensor * squared_diffs

    return torch.mean(weighted_squared_diffs)


def lap_loss(embedding, L):
    """
    Manifold regularization term  for multivariate regression task.
    Args:
        embedding: embedded output from the model. (Batch, N, d)
        L: graph Laplacian (Batch, N, N)

    Returns:
        loss value.
    """
    # Apply Laplacian and compute trace for each sample in the batch
    intermediate = torch.bmm(torch.bmm(embedding.transpose(1, 2), L), embedding)
    loss = torch.trace(intermediate.sum(dim=0))
    # Normalize the loss by the batch size and number of nodes
    batch_size, num_nodes = embedding.shape[0], embedding.shape[1]
    normalized_loss = loss / (batch_size * num_nodes)
    return normalized_loss


def penalty_mask(mask):
    """
    penalty term to constrain the orthonormality of learned mask
    Args:
        mask: embedded output from the model. (N, N)
    Returns:
        penalty loss value.
    """
    # Calculate the deviation of mask * mask.T from the identity matrix
    penalty = torch.norm(torch.matmul(mask, mask.T) - torch.eye(mask.size(0), device=mask.device, dtype=mask.dtype),
                         p='fro')

    return penalty


def classification_loss(output, target):
    """
    Cross Entropy Loss for classification task.
    Args:
        output: Predicted output from the model.
        target: Ground truth.
    Returns:
        Loss value.
    """
    return F.cross_entropy(output, target)

