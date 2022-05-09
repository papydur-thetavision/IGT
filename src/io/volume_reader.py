import math
import os
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import scipy.io as scio
import torch.nn
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from torch.utils.data import Dataset

from src.config import Config


class VolumeReader(Dataset):

    def __init__(self,  data_transform=None):

        self.config = Config()
        self.data_path = self.config.data_path
        self.volume_list = None
        self.normalization_constants = {'mean': 0.134014, 'std': 0.176344}
        self.data_transform = data_transform
        self.endpoints = []

    def precompute_endpoints(self):
        for index in range(len(self.volume_list)):
            data = self.read(index)
            volume, mask = self.normalize_inputs(data)
            volume_shape = torch.Tensor(volume.shape)
            points = self.get_end_points(mask)

            # always set point closest to origin as coord 1
            min_index = points.sum(axis=1).argmin()
            self.endpoints.append({'coord1': torch.Tensor(points[min_index]).float() / volume_shape,
                 'coord2': torch.Tensor(points[1 - min_index]).float() / volume_shape})

    def create_volume_list(self):
        self.volume_list = [volume for volume in os.listdir(self.data_path) if volume.endswith('.mat')]

    def normalize_inputs(self, data):
        volume = data['Volume255'] / 255
        volume = volume - self.normalization_constants['mean'] / self.normalization_constants['std']

        mask = data['Mask'] / 1.0
        return volume, mask

    def read(self, index):
        filename = self.data_path + self.volume_list[index]
        data = scio.loadmat(filename)
        return data

    def __getitem__(self, index):
        data = self.read(index)
        volume, mask = self.normalize_inputs(data)

        if self.endpoints:
            y = self.endpoints[index]
        else:
            volume_shape = torch.Tensor(volume.shape)
            points = self.get_end_points(mask)

            #always set point closest to origin as coord 1
            min_index = points.sum(axis=1).argmin()
            y = {'coord1': torch.Tensor(points[min_index]).float() / volume_shape,
                 'coord2': torch.Tensor(points[1-min_index]).float() / volume_shape}

        volume = torch.Tensor(volume)
        volume = volume.unsqueeze(0).float()

        return volume, y

    def __len__(self):
        if self.volume_list:
            return len(self.volume_list)
        else:
            print('Define volume list first')

    @staticmethod
    def skeletonize_mask(mask):
        mask = mask.round()
        skeleton = skeletonize(mask)
        return skeleton / 255.

    def create_2_planes(self, mask):
        y_1 = self.skeletonize_mask(np.amax(mask, axis=2))
        y_2 = self.skeletonize_mask(np.amax(np.transpose(mask, [2, 1, 0]), axis=2))
        return y_1, y_2

    @staticmethod
    def get_end_point_from_skeleton(skeleton):
        k = np.ones((3, 3, 3))
        k[1, 1, 1] = 10
        conv_skel = convolve(skeleton, k, mode='constant', cval=0)
        np_coords = np.asarray(conv_skel == 11).nonzero()
        points = np.array([np.array(p) for p in zip(*np_coords)])
        return points

    def get_end_points(self, mask):
        skeleton = self.skeletonize_mask(mask)
        points = self.get_end_point_from_skeleton(skeleton)
        return points

    def get_train_val_reader(self, train_percent=.8):
        n_train = math.floor(len(self.volume_list)*train_percent)

        vr_train = VolumeReader(data_transform=self.data_transform)
        vr_train.volume_list = self.volume_list[:n_train]

        vr_val = VolumeReader(data_transform=self.data_transform)
        vr_val.volume_list = self.volume_list[n_train:]

        if self.endpoints:
            vr_train.endpoints = self.endpoints[:n_train]
            vr_val.endpoints = self.endpoints[n_train:]

        return vr_train, vr_val


