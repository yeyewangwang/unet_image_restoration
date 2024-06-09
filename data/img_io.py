import numpy as np
from PIL import Image
import PIL
from typing import List
import torch


def read_and_normalize_rgba_img(img_path: str) -> np.ndarray:
    """
    Load images in (h, w, c) format, normalize rgb values by
    a values, then tranpose image into (c, h, w) format. In the
    end, normalize pixel value range from [0, 255] to [-1, 1].
    """
    rgba_img = np.array(
        (Image.open(img_path)).convert('RGBA'))

    assert rgba_img.ndim == 3, f"Image array must be of dimensinon 3, actual dimension {rgba_img.ndim}"

    r = rgba_img[..., 0]
    g = rgba_img[..., 1]
    b = rgba_img[..., 2]
    norm_a = rgba_img[..., 3] / 255

    img = np.stack([r * norm_a, g * norm_a, b * norm_a], axis=-1)
    norm_img = (img / 127.5 - 1).astype(np.float32)
    return np.transpose(norm_img, (2, 0, 1))


def img_batch_tensor_to_pils(norm_img: torch.Tensor) -> Image:
    """
    Convert an image tensor (b, c, h, w) to a list of b Image objects.

    The image tensor is first unnormalized from [-1, 1] floating
    numbers into [0, 255] integers. Then it is permuted into
    (b, h, w, c) format. In turn, each image is individually
    converted into an Image object.
    """
    if norm_img.requires_grad:
        norm_img = norm_img.detach()
    unnorm_batch = ((norm_img + 1) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    im_array = [Image.fromarray(unnorm_batch[i]) for i in range(unnorm_batch.shape[0])]
    return im_array


def mask_batch_tensor_to_pils(mask: torch.Tensor):
    """
    Convert mask tensor from (b, h, w) to a list of b Image objects.
    """
    if mask.requires_grad:
        mask = mask.detach()
    unnorm_batch = (mask * 255).cpu().numpy().astype(np.uint8)
    unnorm_batch = np.squeeze(unnorm_batch, axis=1)
    im_array = [Image.fromarray(unnorm_batch[i]) for i in
                range(unnorm_batch.shape[0])]
    return im_array
