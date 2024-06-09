import argparse
import os
import random
from typing import Tuple
import numpy as np
import torch
import wandb as wb
from model.image_restoration_model import UnetImageRestorationModel
from utils.loss import reconstruction_loss
from data.dataset import create_data_loader
from data.img_visualize import pred_to_imgs


def worker_init_fn(worker_id: int):
    """Dataloader worker initialization function.
    Ensures that worker uses same random sequence to produce
    each batch, given the same global random seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Trainer Class
class Trainer:
    def __init__(self, train_args):
        self.args = train_args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print("WARNING: RNG state does not persist across " +
              "cuda/Non-cuda devices. " +
              "cuda may opt to use randomness at runtime. "
              "THIS WARNING WILL NOT BE SHOWN LATER!")

        global_seed = 7
        self.global_seed = global_seed
        # Set the seed after loading the checkpoint to ensure consistency
        g = torch.Generator()
        g.manual_seed(global_seed)
        self.set_seed(global_seed, device=self.device)

        train_dir = os.path.join(args.dataset_dir, "train")
        self.train_loader = create_data_loader(train_dir,
                                               batch_size=self.args.batch_size, worker_init_fn=worker_init_fn,
                                               rng=g)

        small_valid_dir = os.path.join(args.dataset_dir, "original_validation")
        self.original_val_loader = create_data_loader(
            small_valid_dir, batch_size=25, is_validation=True, rng=g)

        valid_dir = os.path.join(args.dataset_dir,
                                 "validation")
        self.val_loader = create_data_loader(
            valid_dir, batch_size=512, is_validation=True, rng=g)

        self.model = self.load_model().to(self.device)
        self.criterion = torch.nn.BCELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr)
        wb.init(project="unet-image-restoration")

    @staticmethod
    def load_model():
        """Load the PyTorch model object"""
        return UnetImageRestorationModel()

    @staticmethod
    def set_seed(seed: int,
                 device: str = "cuda",
                 cuda_deterministic: bool = False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(
                seed)  # for multi-GPU
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


    @staticmethod
    def log_imgs(data: torch.Tensor, src_imgs: torch.Tensor,
                 target_mask: torch.Tensor,
                 pred_imgs: torch.Tensor, pred_mask: torch.Tensor,
                 binary_mask_threshold: float,
                 mode="training"):
        img_titles = ["corrupted", "source", "true_mask",
                  "reconstructed", "predicted_image", "predicted_mask"]
        for i, imgs in enumerate(pred_to_imgs(data, src_imgs, target_mask,
                                 pred_imgs, pred_mask,
                                 binary_mask_threshold)):
            wb.log({f"{mode}_{img_titles[i]}": [wb.Image(image) for image
                                in imgs]})

    def save_inference_checkpoint(self, epoch: int, batch_idx: int):
        # Save all necessary components to resume training
        torch.save(self.model.state_dict(),
                   os.path.join(self.args.checkpoint_dir,
                        f'inference_checkpoint_{epoch}_{batch_idx}.pt'))

    def save_checkpoint(self, epoch: int, batch_idx: int, global_seed: int):
        # Save all necessary components to resume training
        torch.save({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if self.device == "cuda" else "NA",
            'np_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            "global_seed": global_seed
        }, os.path.join(self.args.checkpoint_dir,
                        f'checkpoint_{epoch}_{batch_idx}.pt'))

    def load_checkpoint(self, path: str) -> Tuple[int, int, int]:
        checkpoint = torch.load(path)
        self.model.load_state_dict(
            checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        if self.device == "cuda":
            torch.cuda.set_rng_state(
                checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['random_rng_state'])
        global_seed = checkpoint['global_seed']
        return checkpoint['epoch'], checkpoint['batch_idx'], global_seed

    def run_validation(self, use_original: bool = False) -> Tuple[torch.Tensor, float, torch.Tensor]:
        self.model.eval()
        val_loss = 0
        val_loss_mask = 0
        val_loss_l1_images = 0
        val_loader = self.original_val_loader if use_original else self.val_loader
        with torch.no_grad():
            for data, src_img, target_mask in val_loader:
                data = data.to(self.device)
                src_img = src_img.to(self.device)
                target_mask = target_mask.to(self.device)
                reconstruction, mask = self.model(data)
                loss_mask = self.criterion(
                    torch.flatten(mask, start_dim=1),
                    torch.flatten(target_mask,
                                  start_dim=1).to(
                        torch.float32))
                loss_l1_images = reconstruction_loss(
                    reconstruction, src_img, target_mask)
                losses = loss_mask + 2 * loss_l1_images

                if use_original:
                    self.log_imgs(data, src_img,
                                  target_mask,
                                  reconstruction, mask,
                                  0.1,
                                  mode="original_val")
                val_loss += losses
                val_loss_mask += loss_mask
                val_loss_l1_images += loss_l1_images

        val_loss /= len(val_loader)
        val_loss_mask /= len(val_loader)
        val_loss_l1_images /= len(val_loader)
        return val_loss_mask, val_loss_l1_images, val_loss

    def train(self):
        start_epoch, start_batch = 0, 0
        global_seed = self.global_seed
        if self.args.resume:
            start_epoch, start_batch, global_seed = self.load_checkpoint(
                self.args.resume_path)

        self.model.train()
        for epoch in range(start_epoch, self.args.epochs):
            num_batches = len(self.train_loader)
            for batch_idx, (data, target, target_mask) in enumerate(
                    self.train_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue  # Skip past batches if resuming

                # Logging and checkpointing flags
                do_logging = (batch_idx + 1) % self.args.checkpoint_interval == 0
                is_last_batch = batch_idx + 1 == num_batches

                corrupted_img = data.to(
                    self.device)
                src_img = target.to(self.device)
                target_mask = target_mask.to(self.device)

                self.optimizer.zero_grad()
                reconstruction, mask = self.model(corrupted_img)
                loss_mask = self.criterion(torch.flatten(mask, start_dim=1),
                                           torch.flatten(target_mask, start_dim=1).to(torch.float32))
                loss_l1_images = reconstruction_loss(reconstruction, src_img, target_mask)
                loss = loss_mask + 2 * loss_l1_images
                loss.backward()
                self.optimizer.step()

                # Log to Weights and Biases each epoch
                # intended to show any instabilities.
                wb.log(
                    {'train_loss_mask': loss_mask.item()})
                wb.log({'train_l1_images': loss_l1_images})
                wb.log({'train_loss': loss.item()})

                if do_logging:
                    print(
                        f"{epoch=}, {batch_idx=}, {loss_mask.item()=:.32f}, "
                        f"{loss_l1_images=:.32f}, {loss.item()=:.32f}")
                    self.log_imgs(data, target, target_mask,
                                  reconstruction, mask,
                                  0.1, mode="training")
                    self.save_checkpoint(epoch, batch_idx,
                                         global_seed)

                # Saving inference checkpoint at end of every epoch
                if is_last_batch:
                    self.save_inference_checkpoint(
                        epoch, batch_idx)

                for use_original in [True, False]:
                    # Run the 25 image validation set as frequent as logging
                    if use_original and not do_logging:
                        continue
                    # Run the full validation set at the end of each epoch
                    if not use_original and not is_last_batch:
                        continue

                    val_loss_mask, val_loss_l1_images, val_loss = self.run_validation(use_original=use_original)
                    self.model.train()
                    prefix = "original_" if use_original else ""
                    print(f"{epoch=}, {batch_idx=}, {prefix}{val_loss_mask.item()=:.32f}, "
                        f"{prefix}{val_loss_l1_images=:.32f}, {prefix}{val_loss.item()=:.32f}")

                    wb.log({f'{prefix}val_loss_mask': val_loss_mask.item()})
                    wb.log(
                            {f'{prefix}val_l1_images': val_loss_l1_images})
                    wb.log({f'{prefix}val_loss': val_loss.item()})


    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="PyTorch Unet Trainer")
        parser.add_argument('--batch_size', type=int,
                            default=64,
                            help='Input batch size for training')
        parser.add_argument('--epochs', type=int,
                            default=30,
                            help='Number of epochs to train')
        parser.add_argument('--lr', type=float,
                            default=0.0003,
                            help='Learning rate')
        parser.add_argument('--dataset_dir', type=str,
                            default="./input",
                            help='Path for input dataset')
        parser.add_argument('--checkpoint_dir', type=str,
                            default='./checkpoints',
                            help='Directory to save checkpoints')
        parser.add_argument('--checkpoint_interval',
                            type=int, default=60,
                            help='How many batches to wait before creating a checkpoint '
                                 'and computing validation accuracy for the small dataset.')
        parser.add_argument('--resume', action='store_true',
                            help='Resume training from checkpoint')
        parser.add_argument('--resume_path', type=str,
                            help='Path to the checkpoint to resume from')
        return parser.parse_args()


if __name__ == "__main__":
    args = Trainer.parse_args()
    trainer = Trainer(args)
    trainer.train()
