from src.networks.instrument_3d_model_parts import *


class Instrument3DModel(nn.Module):
    
    def __init__(self, base_channels):
        super().__init__()
        self.base_channels = base_channels

        # Encoder
        self.encoder_block_1 = EncoderBlock(in_channels=1, out_channels=self.base_channels, first_kernel_size=7)
        self.encoder_block_2 = EncoderBlock(in_channels=self.base_channels, out_channels=self.base_channels * 2)
        self.encoder_block_3 = EncoderBlock(in_channels=self.base_channels * 2, out_channels=self.base_channels * 4)
        self.encoder_block_4 = EncoderBlock(in_channels=self.base_channels * 4, out_channels=self.base_channels * 8)
        self.encoder_block_5 = Conv3DINReLU(in_channels=self.base_channels * 8, out_channels=self.base_channels * 8)
        self.coord_head = CoordinateHead(in_channels=self.base_channels*8)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder_block_1(x)
        x = self.encoder_block_2(x)
        x = self.encoder_block_3(x)
        x = self.encoder_block_4(x)
        x = self.encoder_block_5(x)
        out = self.coord_head(x)
        return out

