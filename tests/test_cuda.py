import torch


def test_cuda_available():
    assert torch.cuda.is_available()


def test_m4000():
    assert torch.cuda.get_device_name(0) == 'Quadro M4000'

