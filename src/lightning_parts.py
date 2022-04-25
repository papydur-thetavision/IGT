import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.io.volume_reader import VolumeReader
from src.networks.instrument_3d_model import Instrument3DModel


class CatheterDataModule(pl.LightningDataModule):

    def __init__(self, volume_reader: VolumeReader, batch_size: int = 16):
        super().__init__()
        self.volume_reader = volume_reader
        self.volume_reader.create_volume_list()
        self.batch_size = batch_size
        self.vrs = {'train': (self.volume_reader.get_train_val_reader())[0],
                    'val': (self.volume_reader.get_train_val_reader())[1]}

    def train_dataloader(self):
        return DataLoader(self.vrs['train'], batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.vrs['val'], batch_size=self.batch_size, num_workers=2, pin_memory=True)


class Catheter3DSystem(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = Instrument3DModel(base_channels=12)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        '''
        Args:
            batch: {x: input_tensor, y: 2 3d_coordinates}
        Returns: the computed loss
        '''
        # calculate the losses of the two coordinate pairs
        loss = self._calculate_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._calculate_loss(batch)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self._calculate_loss(batch)
        self.log("test_loss", test_loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred

    def _calculate_loss(self, batch):
        x, y = batch
        y_hat = self(x)

        # calculate the losses of the two coordinate pairs
        loss = F.smooth_l1_loss(y_hat['coord1'], y['coord1']) + F.smooth_l1_loss(y_hat['coord2'], y['coord2'])

        return loss




