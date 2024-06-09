import torch
from torch import Tensor


def reconstruction_loss(predicted: Tensor,
                        target: Tensor,
                        target_mask: Tensor,
                        reduce: str = "mean") -> torch.Tensor:
    """
    Args:
        predicted: (B, C, H, W)
        target: (B, C, H, W)
        target_mask: (B, H, W)
    """
    # (b, h, w) -> (b, 1, h, w)
    target_mask = target_mask.unsqueeze(1)

    # Broadcasted element-wise product, with target_mask
    # repeating across dimension indexed 1.
    masked_pred = predicted * target_mask
    masked_target = target * target_mask
    num_channels = predicted.shape[1]

    # since target_mask dimension 1 has size 1,
    # summing across dimensions 2, 3 should give same result
    mask_size = target_mask.sum(dim=[1, 2, 3])

    # PyTorch does will average out the loss across unmarked regions
    # builtin_loss = torch.nn.functional.l1_loss(masked_pred, masked_target, reduction='none')

    eps = 1e-15
    loss = torch.abs(masked_pred - masked_target).sum(dim=[1, 2, 3]) / (mask_size * num_channels + eps)
    if reduce == "mean":
        return loss.mean()
    if reduce == "none":
        return loss

