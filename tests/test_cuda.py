import torch
from src.config import Config


def test_cuda_available():
    assert torch.cuda.is_available()


def test_device():
    config = Config()

    if config.location == 'philips':
        assert torch.cuda.get_device_name(0) == 'Quadro M4000'
    elif config.location == 'uni':
        assert torch.cuda.get_device_name(0) == 'NVIDIA TITAN Xp'
