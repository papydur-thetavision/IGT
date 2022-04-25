import torch


def test_constructor(volume_reader):
    assert volume_reader


def test_mat_files_in_volume_list(volume_reader):
    for volume in volume_reader.volume_list:
        assert volume.endswith('.mat')


def test_read_volume(volume_and_mask):
    volume, mask = volume_and_mask
    assert len(volume.shape) == 3
    assert len(volume.shape) == 3


def test_endpoints_3d(points):
    assert len(points[0]) == 3


def test_all_volumes_have_two_endpoints(volume_reader):
    for index in range(len(volume_reader)):
        data = volume_reader.read(index)
        volume, mask = volume_reader.normalize_inputs(data)
        points = volume_reader.get_end_points(mask)
        assert len(points) == 2


def test_points_float_between_zero_and_one(volume_reader):
    x, y = volume_reader[0]
    for point in y.values():
        for axis in point:
            assert isinstance(axis.item(), float)
            assert axis.item() >= 0
            assert axis.item() <= 1


def test_catheter_data_module(catheter_data_module):
    vr_train = catheter_data_module.vrs['train']
    vr_val = catheter_data_module.vrs['val']

    assert len(vr_train) > len(vr_val)
    assert not (set(vr_train.volume_list) & set(vr_val.volume_list))



