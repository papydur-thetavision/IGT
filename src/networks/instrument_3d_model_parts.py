import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_kernel_size=3):
        super().__init__()
        self.encoder_block = nn.Sequential(
            Conv3DINReLU(in_channels, out_channels, kernel_size=first_kernel_size),
            Conv3DINReLU(out_channels, out_channels, kernel_size=3),
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.encoder_block(x)
        x = self.pool(x)
        return x


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
        return out
