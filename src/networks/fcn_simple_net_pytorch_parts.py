import torch
from torch import nn


class Conv3DReluIN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv3d_relu_bn = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(),
            nn.InstanceNorm3d(out_channels),
        )

    def forward(self, x):
        return self.conv3d_relu_bn(x)


class Conv3DINReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv3d_in_relu = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding='same'),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv3d_in_relu(x)


class Conv2DINReLU(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv2d_bn_relu(x)


class Conv2DReLUIN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv2d_bn_relu(x)


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, first_kernel_size=3):
        super().__init__()
        self.encoder_block = nn.Sequential(
            Conv3DINReLU(in_channels, out_channels, kernel_size=first_kernel_size),
            Conv3DReluIN(out_channels, out_channels, kernel_size=3),
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x1 = self.encoder_block(x)
        x1_pool = self.pool(x1)
        return x1, x1_pool


class ResBlock2D(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv2DReLUIN(in_channels=channels, out_channels=channels),
            Conv2DReLUIN(in_channels=channels, out_channels=channels),
            Conv2DReLUIN(in_channels=channels, out_channels=channels)
        )

    def forward(self, x):
        return self.conv_block(x) + x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.decoder_block = nn.Sequential(
            Conv2DINReLU(in_channels, out_channels),
            Conv2DReLUIN(out_channels, out_channels)
        )

    def forward(self, x):
        x_up = self.up(x)
        return self.decoder_block(x_up)


class DecoderBlockAdd(DecoderBlock):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, x, x_add):
        x_up = self.up(x) + x_add
        return self.decoder_block(x_up)


class DimensionDown(nn.Module):
    def __init__(self, channels, image_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2 * channels, out_channels=channels, kernel_size=3, padding='same')
        self.conv3d = nn.Conv3d(in_channels=channels, out_channels=1, kernel_size=3, padding='same')
        self.conv2d_1 = Conv2DReLUIN(in_channels=image_size, out_channels=1)
        self.conv2d_2 = Conv2DReLUIN(in_channels=1, out_channels=channels)

    def forward(self, x):
        x_max = torch.max(x, 4).values
        x_mean = torch.mean(x, 4)
        x_cat = torch.cat([x_max, x_mean], dim=1)  # add along the  channel dimension
        x_cat = self.conv(x_cat)

        x = self.conv3d(x)
        x = x.squeeze(dim=1)
        x = self.conv2d_1(x)
        return self.conv2d_2(x) + x_cat


class DimensionDownModule(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()
        self.dimension_down = DimensionDown(channels=channels, image_size=image_size)

    def forward(self, x):
        x_t = torch.transpose(x, dim0=2, dim1=4)
        x = self.dimension_down(x)
        x_t = self.dimension_down(x_t)
        return x, x_t


class ContextBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.context_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.context_block(x)


class CoordinateHead(nn.Module):

    '''
    Module that takes a 3d volume and returns 2 3d coordinates
    where the values are between 0 and 1'''

    def __init__(self, in_channels):
        super().__init__()
        self.aap_3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.coord1 = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.LeakyReLU(),
            nn.Linear(in_channels//2, in_channels//4),
            nn.LeakyReLU(),
            nn.Linear(in_channels // 4, 3),
        )

        self.coord2 = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.LeakyReLU(),
            nn.Linear(in_channels//2, in_channels//4),
            nn.LeakyReLU(),
            nn.Linear(in_channels // 4, 3),
        )

    def forward(self, x):
        out = {}
        x = self.aap_3d(x)
        out['coord1'] = self.coord1(x.squeeze())
        out['coord2'] = self.coord2(x.squeeze())
        return x.squeeze()


