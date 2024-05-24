import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResnetBlock, self).__init__()
        self.conv_layer1 = nn.Conv2d(channels,
                                     channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode="reflect")
        self.batch_norm1 = nn.BatchNorm2d(num_features=channels)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        identity = data
        out = self.conv_layer1(data)
        out = self.batch_norm1(out)
        # Return silu(identity + batch norm output)
        return F.silu(identity + out)


class UnetChannelResamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetChannelResamplingBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode="reflect")

    def forward(self, data: torch.Tensor):
        out = self.conv_layer(data)
        return F.silu(out)


class UnetEncoderBlock(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       is_final_block: bool =  False):
        super(UnetEncoderBlock, self).__init__()
        self.resampling_block = \
            UnetChannelResamplingBlock(in_channels, out_channels)
        self.resnet1 = ResnetBlock(out_channels)
        self.resnet2 = ResnetBlock(out_channels)
        self.resnet3 = ResnetBlock(out_channels)
        self.is_final_block = is_final_block

    def forward(self, data: torch.Tensor):
        out = self.resampling_block(data)
        out = self.resnet1(out)
        out = self.resnet2(out)
        skip_connection = self.resnet3(out)
        # (b, c, w, h) -> (b, c, w//2, h//2)
        if self.is_final_block:
            return skip_connection, None
        else:
            out = F.max_pool2d(skip_connection, kernel_size=2)
            return out, skip_connection


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDecoderBlock, self).__init__()
        self.channel_downsampler = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)

        self.resampling_block = UnetChannelResamplingBlock(in_channels, out_channels)

        self.resnet1 = ResnetBlock(out_channels)
        self.resnet2 = ResnetBlock(out_channels)
        self.resnet3 = ResnetBlock(out_channels)

    def forward(self, data: torch.Tensor,
                      skip_connection: torch.Tensor):
        """
        data: (b, c, w, h)
        skip_connection: (b, c//2, 2*w, 2*h)
        """
        assert data.shape[1] == 2*skip_connection.shape[1], \
            f"data channels = {data.shape[1]}, output channels = {skip_connection.shape[1]}, in_channels should be twice out_channels"
        # Handle skip connection
        # interpolate to double spatial dimension size.
        # Use skip_connection.shape[2:] here because
        # Sometimes we need to interpolate from 10 to 21.
        out = F.interpolate(data,
                            size=skip_connection.shape[2:],
                            mode="bilinear",
                            align_corners=True)
        out = self.channel_downsampler(out)

        out = torch.concat((skip_connection, out), dim=1)
        # Decrease number of channels
        out = self.resampling_block(out)
        # Resnet blocks
        out = self.resnet1(out)
        out = self.resnet2(out)
        out = self.resnet3(out)
        return out


class MaskHead(nn.Module):
    def __init__(self, in_channels: int):
        super(MaskHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              padding_mode="reflect")

    def forward(self, data: torch.Tensor):
        out = self.conv(data)
        return F.sigmoid(out)


class ReconstructHead(nn.Module):
    def __init__(self, in_channels: int):
        super(ReconstructHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3,
                              kernel_size=3,
                              stride=1,
                              padding=0)

    def forward(self, data: torch.Tensor):
        out = self.conv(data)
        return F.tanh(out)
