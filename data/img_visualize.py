from typing import List, Tuple, Any
import torch
import numpy as np
from data.img_utils import (convert_img_ndarray_to_pil_img,
                            convert_mask_ndarray_to_pil_img)


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

    # Move to CPU and numpy
    norm_corrupted_img = data[selected_imgs].detach().cpu().numpy()  # 25 x c x h x w
    predicted_img = pred_imgs[selected_imgs].detach().cpu().numpy()  # 25 x c x h x w
    predicted_mask = pred_mask[selected_imgs].detach().cpu().numpy()  # 25 x 1 x h x w
    src_img = src_imgs[selected_imgs].detach().cpu().numpy()  # 25 x c x h x w
    true_mask = target_mask[selected_imgs].detach().cpu().numpy()  # 25 x c x h x w

    # Convert the predicted mask to a binary mask, based on the threshold
    binary_mask = (predicted_mask >= binary_mask_threshold).astype(int)

    # Within the binary mask, the 1s indicate corrupted pixels and 0s indicate
    # uncorrupted pixels. For all the pixels with value 1, copy the "fixed" value
    # from the `predicted_img`. For all the pixels with value 0, copy the "original"
    # from the `norm_corrupted_img`.
    # Match dims of predicted_img (b x 1 x h x w -> b x 3 x h x w)
    expanded_binary_mask = np.repeat(binary_mask, repeats=3, axis=1)
    reconstructed_img = predicted_img * expanded_binary_mask + (
                1 - expanded_binary_mask) * norm_corrupted_img

    # Let's un-normalize the images and masks
    input_pil_imgs = []
    src_pil_imgs = []
    true_binary_mask_pil_imgs = []
    reconstructed_pil_imgs = []
    predicted_pil_imgs = []
    binary_mask_pil_imgs = []

    for i in range(norm_corrupted_img.shape[0]):
        # TODO: use Tensor operations from my own code
        input_pil_imgs.append(convert_img_ndarray_to_pil_img(
                norm_img=norm_corrupted_img[i]))
        src_pil_imgs.append(
            convert_img_ndarray_to_pil_img(
                norm_img=src_img[i]))
        true_binary_mask_pil_imgs.append(
            convert_mask_ndarray_to_pil_img(
                mask=true_mask[i]))
        reconstructed_pil_imgs.append(convert_img_ndarray_to_pil_img(
            norm_img=reconstructed_img[i]))
        predicted_pil_imgs.append(convert_img_ndarray_to_pil_img(
                norm_img=predicted_img[i]))
        binary_mask_pil_imgs.append(convert_mask_ndarray_to_pil_img(
                mask=np.squeeze(binary_mask[i], axis=0)))
    return (input_pil_imgs, src_pil_imgs, true_binary_mask_pil_imgs,
            reconstructed_pil_imgs, predicted_pil_imgs, binary_mask_pil_imgs)
