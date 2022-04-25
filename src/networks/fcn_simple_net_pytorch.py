import winnt

from src.networks.fcn_simple_net_pytorch_parts import *


BASE_CHANNELS = 12


class FcnSimpleNetPytorch(nn.Module):
    """
    pytorch implementation of Hongxu's fcnsimplenet model from tensorflow
    :param image_size: size of the images that will be fed to the model. Required during initialization.
    """
    def __init__(self, image_size=160):
        super().__init__()

        self.activation_map_shapes = dict()
        self.output_dict = {'axial': dict(), 'side': dict()}

        # Encoder
        self.encoder_block_1 = EncoderBlock(in_channels=1, out_channels=BASE_CHANNELS, first_kernel_size=7)
        self.encoder_block_2 = EncoderBlock(in_channels=BASE_CHANNELS, out_channels=BASE_CHANNELS*2)
        self.encoder_block_3 = EncoderBlock(in_channels=BASE_CHANNELS*2, out_channels=BASE_CHANNELS*4)
        self.encoder_block_4 = EncoderBlock(in_channels=BASE_CHANNELS*4, out_channels=BASE_CHANNELS*8)
        self.encoder_block_5 = Conv3DReluIN(in_channels=BASE_CHANNELS*8, out_channels=BASE_CHANNELS*8)

        # Dimensionality reduction
        self.dim_block_1 = DimensionDownModule(channels=2*BASE_CHANNELS, image_size=image_size//2)
        self.dim_block_2 = DimensionDownModule(channels=4*BASE_CHANNELS, image_size=image_size//4)
        self.dim_block_3 = DimensionDownModule(channels=8*BASE_CHANNELS, image_size=image_size//8)
        self.dim_block_4 = DimensionDownModule(channels=8*BASE_CHANNELS, image_size=image_size//16)

        # Decoder
        self.decoder_block_1 = ResBlock2D(channels=8*BASE_CHANNELS)
        self.decoder_block_2 = DecoderBlockAdd(in_channels=8*BASE_CHANNELS, out_channels=4*BASE_CHANNELS)
        self.decoder_block_3 = DecoderBlockAdd(in_channels=4*BASE_CHANNELS, out_channels=2*BASE_CHANNELS)
        self.decoder_block_4 = DecoderBlockAdd(in_channels=2*BASE_CHANNELS, out_channels=2*BASE_CHANNELS)
        self.decoder_block_5 = DecoderBlock(in_channels=2*BASE_CHANNELS, out_channels=BASE_CHANNELS)

        # Deep supervisions upsample
        self.ds_up_1 = nn.ConvTranspose2d(in_channels=4*BASE_CHANNELS, out_channels=1,
                                          kernel_size=8, stride=8)
        self.ds_up_2 = nn.ConvTranspose2d(in_channels=2*BASE_CHANNELS,
                                          out_channels=1, kernel_size=4, stride=4)
        self.ds_up_3 = nn.ConvTranspose2d(in_channels=2*BASE_CHANNELS,
                                          out_channels=1, kernel_size=2, stride=2)

        # output
        self.out_conv = nn.Conv2d(in_channels=BASE_CHANNELS, out_channels=1, kernel_size=3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # encoder forward pass
        x1, x1_pool = self.encoder_block_1(x)
        x2, x2_pool = self.encoder_block_2(x1_pool)
        x3, x3_pool = self.encoder_block_3(x2_pool)
        x4, x4_pool = self.encoder_block_4(x3_pool)
        x5 = self.encoder_block_5(x4_pool)

        # Dimensionality reduction
        x2_d_1, x2_d_2 = self.dim_block_1(x2)
        x3_d_1, x3_d_2 = self.dim_block_2(x3)
        x4_d_1, x4_d_2 = self.dim_block_3(x4)
        x5_d_1, x5_d_2 = self.dim_block_4(x5)


        # Decoder
        x_1 = self.decoder_block_1(x5_d_1)
        x_1 = self.decoder_block_2(x_1, x4_d_1)
        x_1_ds1 = self.ds_up_1(x_1)
        x_1 = self.decoder_block_3(x_1, x3_d_1)
        x_1_ds2 = self.ds_up_2(x_1)
        x_1 = self.decoder_block_4(x_1, x2_d_1)
        x_1_ds3 = self.ds_up_3(x_1)
        x_1 = self.decoder_block_5(x_1)
        x_1 = self.out_conv(x_1)

        x_2 = self.decoder_block_1(x5_d_2)
        x_2 = self.decoder_block_2(x_2, x4_d_2)
        x_2_ds1 = self.ds_up_1(x_2)
        x_2 = self.decoder_block_3(x_2, x3_d_2)
        x_2_ds2 = self.ds_up_2(x_2)
        x_2 = self.decoder_block_4(x_2, x2_d_2)
        x_2_ds3 = self.ds_up_3(x_2)
        x_2 = self.decoder_block_5(x_2)
        x_2 = self.out_conv(x_2)

        self.output_dict['axial'] = {
            'output': self.sigmoid(x_1),
            'up1': self.sigmoid(x_1_ds1),
            'up2': self.sigmoid(x_1_ds2),
            'up3': self.sigmoid(x_1_ds3)
        }
        self.output_dict['side'] = {
            'output': self.sigmoid(x_2),
            'up1': self.sigmoid(x_2_ds1),
            'up2': self.sigmoid(x_2_ds2),
            'up3': self.sigmoid(x_2_ds3)
        }

        return self.output_dict


class ContextEncoder:

    def __init__(self, channels):
        self.block1 = ContextBlock(in_channels=1, out_channels=channels)
        self.block2 = ContextBlock(in_channels=channels, out_channels=2*channels)
        self.block3 = ContextBlock(in_channels=2*channels, out_channels=4*channels)
        self.block4 = ContextBlock(in_channels=4*channels, out_channels=8*channels)
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=8*channels, out_channels=8*channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(8*channels)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.block5(x)



