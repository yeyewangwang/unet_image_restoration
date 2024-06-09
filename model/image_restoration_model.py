"""
U-Net for Image Restoration Model, with 1 encoder
and 2 completely separate decoders.
"""
from typing import Tuple
import torch
import torch.nn as nn
from model.unet_blocks import UnetEncoderBlock, UnetDecoderBlock, MaskHead, ReconstructHead


class UnetImageRestorationModel(nn.Module):
    def __init__(self):
        super(UnetImageRestorationModel, self).__init__()
        init_channels = 32
        self.enc1 = UnetEncoderBlock(in_channels=3,
                                     out_channels=init_channels)
        self.enc2 = UnetEncoderBlock(in_channels=init_channels,
                                     out_channels=2*init_channels)
        self.enc3 = UnetEncoderBlock(in_channels=2*init_channels,
                                     out_channels=4*init_channels)
        self.enc4 = UnetEncoderBlock(in_channels=4*init_channels,
                                     out_channels=8*init_channels)
        self.enc5 = UnetEncoderBlock(in_channels=8*init_channels,
                                     out_channels=16*init_channels,
                                     is_final_block=True)
        self.dec4 = UnetDecoderBlock(
            in_channels=16*init_channels,
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
            in_channels=16*init_channels,
            out_channels=8*init_channels)
        self.mask_dec3 = UnetDecoderBlock(
            in_channels=8*init_channels,
            out_channels=4*init_channels)
        self.mask_dec2 = UnetDecoderBlock(
            in_channels=4*init_channels,
            out_channels=2*init_channels)
        self.mask_dec1 = UnetDecoderBlock(
            in_channels=2*init_channels,
            out_channels=init_channels)
        self.reconstruction_head = ReconstructHead(in_channels=init_channels)
        self.mask_head = MaskHead(in_channels=init_channels)

    def forward(
            self, corrupted_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate a forward pass of the Unet Image Restoration Model.

        Args:
            corrupted_image: (b, c, h, w), b = batch_size, c = # channels,
            h = image height, w = image width. Values should be normalized
            between -1 and 1.

        Outputs:
            a tuple of (predicted_image, predicted_binary_mask)
            predicted_image: (b, c, h, w) output of the image decoder, this
            is the full predicted where each pixel is predicted, NOT
            the reconstructed image.

            predicted_binary_mask: (b, 1, h, w) output of the binary
            mask decoder. Should be a probability value indicating
            the chance of a pixel being corrupted.
        """
        # print(corrupted_image.size())
        out, skip1 = self.enc1(corrupted_image)
        # print(out.size())
        out, skip2 = self.enc2(out)
        # print(out.size())
        out, skip3 = self.enc3(out)
        # print(out.size())
        out, skip4 = self.enc4(out)
        # print(out.size())
        out, _ = self.enc5(out)
        encoder_out = out
        # print(out.size(), skip4.size())
        out = self.dec4(out, skip4)
        # print(out.size(), skip3.size())
        out = self.dec3(out, skip3)
        # print(out.size(), skip2.size())
        out = self.dec2(out, skip2)
        # print(out.size(), skip1.size())
        out = self.dec1(out, skip1)
        # print(encoder_out.size(), skip4.size())
        mask_out = self.mask_dec4(encoder_out, skip4)
        # print(mask_out.size(), skip3.size())
        mask_out = self.mask_dec3(mask_out, skip3)
        # print(mask_out.size(), skip2.size())
        mask_out = self.mask_dec2(mask_out, skip2)
        # print(mask_out.size(), skip1.size())
        mask_out = self.mask_dec1(mask_out, skip1)
        # print(out.size())
        return self.reconstruction_head(out),\
               self.mask_head(mask_out)
