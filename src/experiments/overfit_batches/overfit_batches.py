from pytorch_lightning.callbacks import EarlyStopping

from src.lightning_parts import CatheterDataModule, Catheter3DSystem
from src.io.volume_reader import VolumeReader
from src.visualization.explore_volume import plot_from_prediction
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
import torch


def show_example(index=0):
    model = Catheter3DSystem().load_from_checkpoint(
        checkpoint_path='lightning_logs/overfit_debug/checkpoints/epoch=238-step=239.ckpt')
    catheter_data_module = CatheterDataModule(volume_reader=VolumeReader(), batch_size=4)
    input_vol, y = catheter_data_module.vrs['train'][index]

    model.eval()
    with torch.no_grad():
        y_pred = model(input_vol.unsqueeze(0))

    plot_from_prediction(volume=input_vol, y=y, y_pred=y_pred)


def train_overfit_model():
    model = Catheter3DSystem()
    catheter_data_module = CatheterDataModule(volume_reader=VolumeReader(), batch_size=4)

    logger = pl_loggers.TensorBoardLogger(version='overfit_debug')

    trainer = pl.Trainer(logger=logger, overfit_batches=1, accelerator='gpu',
                         devices='1', log_every_n_steps=1,
                         callbacks=[EarlyStopping(monitor='train_loss', verbose=True, patience=20)])
    trainer.fit(model, catheter_data_module)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('current device: {}'.format(torch.cuda.get_device_name(0)))
    show_example(1)





