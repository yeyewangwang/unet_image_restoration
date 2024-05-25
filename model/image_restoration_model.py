"""
File: image_restoration_model.py
Description: Image Restoration Model (U-NET with 1 Encoder and 2 Decoders)
"""
from typing import Tuple
import torch
import torch.nn as nn
from model.unet_blocks import UnetEncoderBlock, UnetDecoderBlock, MaskHead, ReconstructHead


class ImageRestorationModel(nn.Module):
    def __init__(self):
        super(ImageRestorationModel, self).__init__()
        init_channels = 32
        self.enc1 = UnetEncoderBlock(in_channels=3, out_channels=init_channels)
        self.enc2 = UnetEncoderBlock(in_channels=init_channels, out_channels=2*init_channels)
        self.enc3 = UnetEncoderBlock(in_channels=2*init_channels, out_channels=4*init_channels)
        self.enc4 = UnetEncoderBlock(in_channels=4*init_channels, out_channels=8*init_channels)
        self.enc5 = UnetEncoderBlock(in_channels=8*init_channels, out_channels=16*init_channels, is_final_block=True)
        self.dec4 = UnetDecoderBlock(
            in_channels=16* init_channels,
            out_channels=8*init_channels)
        self.dec3 = UnetDecoderBlock(
            in_channels=8*init_channels,
            out_channels=4*init_channels)
        self.dec2 = UnetDecoderBlock(
            in_channels=4*init_channels,
            out_channels=2*init_channels)
        self.dec1 = UnetDecoderBlock(
            in_channels=2*init_channels,
            out_channels=init_channels)
        self.mask_dec4 = UnetDecoderBlock(
            in_channels=16 * init_channels,
            out_channels=8 * init_channels)
        self.mask_dec3 = UnetDecoderBlock(
            in_channels=8 * init_channels,
            out_channels=4 * init_channels)
        self.mask_dec2 = UnetDecoderBlock(
            in_channels=4 * init_channels,
            out_channels=2 * init_channels)
        self.mask_dec1 = UnetDecoderBlock(
            in_channels=2 * init_channels,
            out_channels=init_channels)
        self.reconstruction_head = ReconstructHead(in_channels=init_channels)
        self.mask_head = MaskHead(in_channels=init_channels)

    def forward(
            self, corrupted_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Image Restoration Model.

        Given a `corrupted_image` with shape (B, C, H, W) where B = batch size, C = # channels,
        H = image height, W = image width and normalized values between -1 and 1,
        run the Image Restoration Model forward and return a tuple of two tensors:
        (`predicted_image`, `predicted_binary_mask`).

        The `predicted_image` should be the output of the Image Decoder (B, C, H, W). In the
        assignment this is referred to as x^{hat}. This is NOT the `reconstructed_image`,
        referred to as `x_{reconstructed}` in the assignment handout.

        The `predicted_binary_mask` should be the output of the Binary Mask Decoder (B, 1, H, W). This
        is `m^{hat}` in the assignment handout.
        """
        # print(corrupted_image.size())
        out, skip1 = self.enc1(corrupted_image)
        # print(out.size(), f"skip1 {skip1.size()}")
        out, skip2 = self.enc2(out)
        # print(out.size(), f"skip2 {skip2.size()}")
        out, skip3 = self.enc3(out)
        # print(out.size(), f"skip3 {skip3.size()}")
        out, skip4 = self.enc4(out)
        # print(out.size(), f"skip4 {skip4.size()}")
        out, _ = self.enc5(out)
        encoder_out = out
        # print(out.size())
        # print(out.size(), skip4.size())
        out = self.dec4(out, skip4)
        # print(out.size())
        out = self.dec3(out, skip3)
        # print(out.size())
        out = self.dec2(out, skip2)
        # print(out.size())
        out = self.dec1(out, skip1)
        # print(out.size())
        mask_out = self.mask_dec4(encoder_out, skip4)
        # print(mask_out.size())
        mask_out = self.mask_dec3(mask_out, skip3)
        # print(mask_out.size())
        mask_out = self.mask_dec2(mask_out, skip2)
        # print(mask_out.size())
        mask_out = self.mask_dec1(mask_out, skip1)
        # print(out.size())
        return self.reconstruction_head(out), self.mask_head(mask_out)
