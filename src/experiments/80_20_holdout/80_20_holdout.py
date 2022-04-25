import os

import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping

from src.io.volume_reader import VolumeReader
from src.lightning_parts import CatheterDataModule, Catheter3DSystem
from src.visualization.explore_volume import plot_from_prediction


def show_example(index=0):
    model = Catheter3DSystem().load_from_checkpoint(
        checkpoint_path='lightning_logs/80_20_holdout/version_1/checkpoints/epoch=31-step=96.ckpt')
    catheter_data_module = CatheterDataModule(volume_reader=VolumeReader(), batch_size=4)
    input_vol, y = catheter_data_module.vrs['val'][index]
    model.eval()
    with torch.no_grad():
        y_pred = model(input_vol.unsqueeze(0))

    plot_from_prediction(volume=input_vol, y=y, y_pred=y_pred)


def train_model():
    model = Catheter3DSystem()
    catheter_data_module = CatheterDataModule(volume_reader=VolumeReader(), batch_size=4)

    logger = pl_loggers.TensorBoardLogger(name='80_20_holdout', version=1, save_dir='lightning_logs/')

    trainer = pl.Trainer(logger=logger, accelerator='gpu', max_epochs=-1, precision=16,
                         devices='1', log_every_n_steps=1, accumulate_grad_batches=8, auto_lr_find=True,
                         callbacks=[EarlyStopping(monitor='val_loss', verbose=True, patience=20)])
    trainer.fit(model, catheter_data_module)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('current device: {}'.format(torch.cuda.get_device_name(0)))
    show_example(index=6)
