import pytest
import torch
import numpy as np

from src.io.volume_reader import VolumeReader
from src.networks.fcn_simple_net_pytorch import FcnSimpleNetPytorch
from src.networks.network_tf import NetWork as FcnSimpleNetTensorflow
from src.lightning_parts import CatheterDataModule, Catheter3DSystem


@pytest.fixture
def volume_reader() -> VolumeReader:
    vr = VolumeReader()
    vr.create_volume_list()
    return vr


@pytest.fixture
def volume_and_mask(volume_reader):
    data = volume_reader.read(index=0)
    volume, mask = volume_reader.normalize_inputs(data)
    return volume, mask


@pytest.fixture()
def points(volume_reader, volume_and_mask):
    skeleton = volume_reader.skeletonize_mask(volume_and_mask[1])
    points = volume_reader.get_end_point_from_skeleton(skeleton)
    return points


@pytest.fixture
def pytorch_model_and_input_160() -> (FcnSimpleNetPytorch, torch.Tensor):
    return FcnSimpleNetPytorch(image_size=160), torch.rand(1, 1, 160, 160, 160)


@pytest.fixture
def tensorflow_model_and_input_160() -> (FcnSimpleNetTensorflow, np.array):
    net = FcnSimpleNetTensorflow()
    test_input = np.random.rand(160, 160, 160)
    return net, test_input


@pytest.fixture
def catheter_data_module(volume_reader):
    catheter_data_module = CatheterDataModule(volume_reader=volume_reader, batch_size=4)
    return catheter_data_module


@pytest.fixture
def catheter_3d_system():
    model = Catheter3DSystem()
    return model


@pytest.fixture
def trained_catheter_3d_system(catheter_3d_system):
    checkpoint = 'epoch=19-step=20.ckpt'
    model = catheter_3d_system.load_from_checkpoint(checkpoint_path=checkpoint)
    return model
