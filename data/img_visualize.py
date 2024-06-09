from typing import List, Tuple, Any
import torch
from data.img_io import (img_batch_tensor_to_pils, mask_batch_tensor_to_pils)


def select_random_indices(loss_tensor: torch.Tensor, num_indices: torch.Tensor) -> torch.Tensor:
    """Randomly select a few indices"""
    indices = torch.randperm(len(loss_tensor))[:num_indices]
    return indices


def pred_to_imgs(data: torch.Tensor, src_imgs: torch.Tensor, target_mask: torch.Tensor,
                 pred_imgs: torch.Tensor, pred_mask: torch.Tensor,
                 binary_mask_threshold: float) -> Tuple[List[Any], List[Any], List[Any],
                                                        List[Any], List[Any], List[Any]]:
    """
    Based on provided code. Take model inputs and outputs,
    return arrays of 25 human readable images.

    Assume that each tensor starts has dimension b as its first dimension
    data: (b, c, h, w), corrupted images in normalized format
    target: (b, c, h, w), source images in normalized format
    target_mask: (b, h, w), true masks
    pred_imgs: (b, c, h, w), predicted full images
    pred_mask: (b, 1, h, w), predicted masks

    Returns a tuple of PIL lists in the following order:
    (a list of corrupted images,a list of source images,
    a list of true masks,
    a list of reconstructed images,a list of predicted images,
    a list of predicted masks).

    Note: this code should be optimized for parallelism on GPU.
    """
    # Testing only
    # selected_imgs = torch.arange(3)
    selected_imgs = torch.arange(25)
    norm_corrupted_img = data[selected_imgs]  # 25 x c x h x w
    predicted_img = pred_imgs[selected_imgs]  # 25 x c x h x w
    predicted_mask = pred_mask[selected_imgs]  # 25 x 1 x h x w
    src_img = src_imgs[selected_imgs]  # 25 x c x h x w
    true_mask = target_mask[selected_imgs]  # 25 x c x h x w

    binary_mask = (predicted_mask >= binary_mask_threshold).int()
    expanded_binary_mask = binary_mask.repeat(1, 3, 1, 1)
    reconstructed_img = predicted_img * expanded_binary_mask + (
                1 - expanded_binary_mask) * norm_corrupted_img

    input_pil_imgs = img_batch_tensor_to_pils(norm_corrupted_img)
    src_pil_imgs = img_batch_tensor_to_pils(src_img)
    true_binary_mask_pil_imgs = mask_batch_tensor_to_pils(true_mask)
    reconstructed_pil_imgs = img_batch_tensor_to_pils(reconstructed_img)
    predicted_pil_imgs = img_batch_tensor_to_pils(predicted_img)
    binary_mask_pil_imgs = mask_batch_tensor_to_pils(binary_mask)

    return (input_pil_imgs, src_pil_imgs, true_binary_mask_pil_imgs,
            reconstructed_pil_imgs, predicted_pil_imgs, binary_mask_pil_imgs)
