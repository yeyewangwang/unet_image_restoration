import os, glob
from typing import Tuple, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Dataset, Subset
from data.img_io import read_and_normalize_rgba_img


class ImageRestorationDataset(Dataset):
    def __init__(self, img_dir: str,
                        transform=None,
                        target_transform=None,
                        mask_transform=None,
                        output_img_id=False,
                        # TODO: resize any input
                        resize: bool = False,
                        inference_only: bool = False):
        self.img_mask_bases = []

        for filename in list(glob.glob(f"{img_dir}/corrupted_imgs/*.png")):
        # for filename in os.listdir(os.path.join(img_dir, "corrupted_imgs")):
            filebase = os.path.basename(filename).split(".")[0]
            self.img_mask_bases.append(filebase)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = transform
        self.mask_transform = mask_transform
        self.output_img_id = output_img_id
        self.inference_only = inference_only

    def __len__(self):
        return len(self.img_mask_bases)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple[np.ndarray,np.ndarray]]:
        img_id = self.img_mask_bases[idx]

        corrupt_img_path = os.path.join(self.img_dir, "corrupted_imgs", f"{img_id}.png")
        corrupt_img = read_and_normalize_rgba_img(corrupt_img_path)

        if not self.inference_only:
            src_img_path = os.path.join(self.img_dir, "src_imgs", f"{img_id}.png")
            # src_img = normalize_rgb_img(read_rgba_img(src_img_path))
            src_img = read_and_normalize_rgba_img(src_img_path)

            mask_path = os.path.join(self.img_dir, "binary_masks", f"{img_id}.npy")
            mask = np.load(mask_path)

            if self.target_transform:
                src_img = self.target_transform(src_img)
            if self.mask_transform:
                mask = self.mask_transform(mask)
        else:
            src_img, mask = np.full(1, np.nan), np.full(1, np.nan)

        if self.transform:
            corrupt_img = self.transform(corrupt_img)

        if self.output_img_id:
            return corrupt_img, src_img, mask, corrupt_img_path
        return corrupt_img, src_img, mask


def create_data_loader(data_dir: str,
                       batch_size: int,
                       is_validation: bool = False,
                       worker_init_fn: Callable = None,
                       rng: torch.Generator = None,
                       output_img_id = False,
                       inference_only = False) -> torch.utils.data.DataLoader:
    """
    Output_img_id: if true, data loader will output the file path to the original id
    as its final output.
    """
    transform = torch.from_numpy
    dataset = ImageRestorationDataset(img_dir=data_dir,
                                      transform=transform,
                                      target_transform=transform,
                                      mask_transform=transform,
                                      output_img_id=output_img_id,
                                      inference_only=inference_only)
    # [Testing only] limit dataloading size for testing purposes
    # dataset = Subset(dataset, range(10))
    # batch_size = min(batch_size, 3)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True if not is_validation else True,
                        num_workers=1,
                        prefetch_factor=1,
                        drop_last=True,
                        worker_init_fn=worker_init_fn,
                        generator=rng)
    return loader
