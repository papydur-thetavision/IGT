import os
import torch


class Config:
    def __init__(self):
        self.location = 'uni'

        if self.location == 'philips':
            self.data_path = 'C:/Users/320181892/Documents/python/Hongxu_IGT/Hongxu_materials/Dataset/'
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            self.num_workers = 8
        elif self.location == 'uni':
            self.data_path = 'F:/IGT/Dataset/'
            self.num_workers = 6

        self.current_device = torch.cuda.get_device_name(0)
        self.holdout_checkpoint = 'lightning_logs/80_20_holdout/v3/checkpoints/epoch=499-step=1500.ckpt'



