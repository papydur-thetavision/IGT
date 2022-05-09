import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from src.io.volume_reader import VolumeReader
from src.lightning_parts import CatheterDataModule, Catheter3DSystem
from src.visualization.explore_volume import plot_from_prediction
from src.config import Config


def show_example(index=0):
    config = Config()
    model = Catheter3DSystem().load_from_checkpoint(
        checkpoint_path=config.holdout_checkpoint)
    catheter_data_module = CatheterDataModule(volume_reader=VolumeReader(), batch_size=4)
    input_vol, y = catheter_data_module.vrs['val'][index]
    model.eval()
    with torch.no_grad():
        y_pred = model(input_vol.unsqueeze(0))

    plot_from_prediction(volume=input_vol, y=y, y_pred=y_pred)


def train_model():
    model = Catheter3DSystem()
    catheter_data_module = CatheterDataModule(volume_reader=VolumeReader(), batch_size=4)

    logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs/', name='80_20_holdout', version='v4')

    trainer = pl.Trainer(logger=logger, accelerator='gpu', max_epochs=-1,
                         devices='1', log_every_n_steps=1, accumulate_grad_batches=8,
                         callbacks=[  # EarlyStopping(monitor='val_loss', verbose=True, patience=20),
                                    LearningRateMonitor(logging_interval='step')
    ])
    trainer.fit(model, catheter_data_module)


def get_val_predictions():
    config = Config()
    model = Catheter3DSystem().load_from_checkpoint(checkpoint_path=config.holdout_checkpoint)
    catheter_data_module = CatheterDataModule(volume_reader=VolumeReader(), batch_size=4)

    trainer = pl.Trainer(accelerator='gpu')
    out = trainer.predict(model, dataloaders=catheter_data_module.train_dataloader())

    return out


if __name__ == '__main__':
    train_model()
