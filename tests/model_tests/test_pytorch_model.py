from src.networks.fcn_simple_net_pytorch import FcnSimpleNetPytorch
from src.networks.network_tf import NetWork
import pytest
import torch


def test_constructor():
    net = FcnSimpleNetPytorch()
    assert net


@pytest.mark.parametrize('image_size', [160, 80])
def test_same_output_size(image_size):
    # GIVEN: an initialized pytorch model and a corresponding input tensor
    net = FcnSimpleNetPytorch(image_size=image_size)
    input_shape = (1, 1, image_size, image_size, image_size)
    test_input = torch.rand(input_shape)
    # WHEN: the tensor is fed to the network
    out = net(test_input)
    # THEN: the output shape of both planes is the same as the input shape 1 dimension lower
    assert out['axial']['output'].shape == input_shape[:-1]
    assert out['side']['output'].shape == input_shape[:-1]


def test_output_range(pytorch_model_and_input_160):
    # GIVEN: an initialized model and input
    net, test_input = pytorch_model_and_input_160
    # WHEN: the input is fed to the network
    out = net(test_input)
    # THEN: the output of both planes are range bound between 0 and 1
    assert out['axial']['output'].flatten().max() < 1.0
    assert out['axial']['output'].flatten().min() > 0

    assert out['side']['output'].flatten().max() < 1.0
    assert out['side']['output'].flatten().min() > 0


def test_deep_supervision_correct_size(pytorch_model_and_input_160):
    # GIVEN: an initialized model and input
    net, test_input = pytorch_model_and_input_160
    # WHEN: the input is fed to the network
    out = net(test_input)
    # THEN: deep supervision outputs same size as output
    output_shape = out['axial']['output'].shape
    assert output_shape == out['side']['output'].shape
    assert output_shape == out['axial']['up1'].shape
    assert output_shape == out['axial']['up2'].shape
    assert output_shape == out['axial']['up3'].shape
    assert output_shape == out['side']['up1'].shape
    assert output_shape == out['side']['up2'].shape
    assert output_shape == out['side']['up3'].shape


@pytest.mark.skip(reason='output of pytorch model not supported')
def test_pytorch_tensorflow_model_shape_equivalence(pytorch_model_and_input_160, tensorflow_model_and_input_160):
    # GIVEN: a pytorch and tensorflow model and an appropriate input
    pytorch_net, pytorch_input = pytorch_model_and_input_160
    tensorflow_net, tensorflow_input = tensorflow_model_and_input_160
    # WHEN: the inputs are fed through the networks
    _ = pytorch_net(pytorch_input)
    _ = tensorflow_net.predict(tensorflow_input)
    # THEN: The shapes of the intermediate layers are equal for both models
    pytorch_shapes = pytorch_net.activation_map_shapes
    tensorflow_shapes = tensorflow_net.activation_map_shapes

    for k in pytorch_shapes.keys():
        # skip batchsize
        pytorch_shape = pytorch_shapes[k][1:]
        tensorflow_shape = tensorflow_shapes[k][1:]
        # transform tensorflow tensor shape to pytorch tensor shape
        tensorflow_shape[0], tensorflow_shape[-1] = tensorflow_shape[-1], tensorflow_shape[0]

        assert pytorch_shape == tensorflow_shape


