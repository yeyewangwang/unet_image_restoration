import torch
import torch.nn as nn
from torch import Tensor


def reconstruction_loss(predicted: Tensor, target: Tensor, target_mask: Tensor):
    """
    predicted: (B, C, H, W)
    target: (B, C, H, W)
    target_mask: (B, H, W)
    """
    target_mask = target_mask.unsqueeze(1)

    masked_pred = predicted * target_mask
    masked_target = target * target_mask
    num_channels = predicted.shape[1]
    mask_size = target_mask.sum(dim=[1, 2, 3])  # (B, 1, 1)
    eps = 1e-15
    loss = torch.abs(masked_pred - masked_target).sum(dim=[1, 2, 3]) / (mask_size * num_channels + eps)
    return loss.mean()

