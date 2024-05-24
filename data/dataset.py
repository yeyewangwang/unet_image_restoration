import os, re
from typing import Tuple
import numpy as np
from torch.utils.data import Dataset
from data.img_utils import read_rgba_img, normalize_rgb_img


class ImageRestorationDataset(Dataset):
    def __init__(self, img_dir: str,
                        transform=None,
                        target_transform=None,
                        mask_transform=None):
        self.img_mask_bases = [
        ]
        for filename in os.listdir(os.path.join(img_dir, "binary_masks")):
            filebase = os.path.basename(filename).split(".")[
                0]
            self.img_mask_bases.append(filebase)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_mask_bases)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple[np.ndarray,np.ndarray]]:
        img_id = self.img_mask_bases[idx]
        # print(f"Image id: {img_id}")

        corrupt_img_path = os.path.join(self.img_dir, "corrupted_imgs", f"{img_id}.png")
        corrupt_img = normalize_rgb_img(read_rgba_img(corrupt_img_path))

        src_img_path = os.path.join(self.img_dir, "src_imgs", f"{img_id}.png")
        src_img = normalize_rgb_img(read_rgba_img(src_img_path))

        mask_path = os.path.join(self.img_dir, "binary_masks", f"{img_id}.npy")
        mask = np.load(mask_path)

        if self.transform:
            corrupt_img = self.transform(corrupt_img)
        if self.target_transform:
            src_img = self.target_transform(src_img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return corrupt_img, src_img, mask
