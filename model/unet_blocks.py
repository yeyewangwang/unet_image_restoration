import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(channels,
                              channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              # this layer is always immediately before
                              # batch normalization. Bias is redundant here.
                              bias=False,
                              padding_mode="reflect")
        self.batch_norm = nn.BatchNorm2d(
            num_features=channels)
        self.silu = torch.nn.SiLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        identity = data
        out = self.conv(data)
        out = self.batch_norm(out)
        return self.silu(identity + out)


class UnetChannelResamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetChannelResamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode="reflect")
        self.silu = torch.nn.SiLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        out = self.conv(data)
        return self.silu(out)


class UnetEncoderBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 is_final_block: bool = False):
        super(UnetEncoderBlock, self).__init__()
        self.is_final_block = is_final_block

        self.channel_upsampler = \
            UnetChannelResamplingBlock(in_channels,
                                       out_channels)
        self.resblock1 = ResBlock(out_channels)
        self.resblock2 = ResBlock(out_channels)
        self.resblock3 = ResBlock(out_channels)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: (b, c, h, w)
        output: (b, 2*c, h, w), (b, 2*c, h//2, w//2)
        """
        out = self.channel_upsampler(
            data)  # (b, c, h, w) -> (b, 2*c, h//2, w//2)
        out = self.resblock1(out)
        out = self.resblock2(out)
        skip_connection = self.resblock3(out)
        if self.is_final_block:
            # skip connection: (b, 2*c, h, w)
            return skip_connection, None
        else:
            # (b, 2*c, h, w) -> (b, 2*c, h//2, w//2)
            out = self.maxpool(skip_connection)
            return out, skip_connection


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDecoderBlock, self).__init__()
        # Downsample the channels to match that of the skip connection
        # Match will be exact encoder always scales it up by 2.
        self.channel_downsampler1 = nn.Conv2d(in_channels,
                                             in_channels // 2,
                                             kernel_size=1)

        # Downsample the channels (input channels stacked with skip connection) to produce output
        self.channel_downsampler2 = UnetChannelResamplingBlock(
            in_channels, out_channels)

        self.resblock1 = ResBlock(out_channels)
        self.resblock2 = ResBlock(out_channels)
        self.resblock3 = ResBlock(out_channels)

    def forward(self, data: torch.Tensor,
                skip_connection: torch.Tensor) -> torch.Tensor:
        """
        data: (b, c, h, w)
        skip_connection: (b, c//2, 2*h + 1, 2*w + 1)
        output: (b, c, 2*h + 1, 2*w + 1)

        Note: skip connection is usually (b, c//2, 2*h, 2*w). Since
        it's trickier when it's (b, c//2, 2*h + 1, 2*w + 1),
        comments will use it as example.
        """
        # Handle skip connection
        # interpolate to double spatial dimension size.
        # Use skip_connection.shape[2:] here because
        # sometimes we need to interpolate from 10 to 21.
        # (b, c, h, w) -> (b, c, 2*h + 1, 2*w + 1)
        out = F.interpolate(data,
                            size=skip_connection.shape[2:],
                            mode="bilinear",
                            align_corners=True)
        # (b, c, 2*h + 1, 2*w + 1) -> (b, c//2, 2*h + 1, 2*w + 1)
        out = self.channel_downsampler1(out)
        # Produce (b, c, 2*h + 1, 2*w + 1)
        out = torch.concat((skip_connection, out), dim=1)
        # Decrease number of channels
        # (b, c, 2*h + 1, 2*w + 1) -> (b, c//2, 2*h + 1, 2*w + 1)
        out = self.channel_downsampler2(out)
        # Resnet blocks
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        return out


class MaskHead(nn.Module):
    def __init__(self, in_channels: int):
        super(MaskHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        out = self.conv(data)
        return self.sigmoid(out)


class ReconstructHead(nn.Module):
    def __init__(self, in_channels: int):
        super(ReconstructHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.tanh = nn.Tanh()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        out = self.conv(data)
        return self.tanh(out)
