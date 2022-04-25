import torch


def test_model_constructor(catheter_3d_system):
    assert catheter_3d_system


def test_same_input_same_output(trained_catheter_3d_system):
    # GIVEN: a model and an input
    rand_input = torch.randn([1, 160, 160, 160])
    # WHEN: input is supplied twice
    with torch.no_grad():
        out1 = trained_catheter_3d_system(rand_input)
        out2 = trained_catheter_3d_system(rand_input)
    # THEN: outputs are the same
    for key in ['coord1', 'coord2']:
        for (item1, item2) in zip(out1[key], out2[key]):
            assert item1 == item2


def test_different_input_different_output(catheter_3d_system):
    # GIVEN: a model and an input
    rand_input1 = torch.randn([1, 160, 160, 160])
    rand_input2 = torch.randn([1, 160, 160, 160])
    # WHEN: input is supplied twice
    with torch.no_grad():
        out1= catheter_3d_system(rand_input1)
        out2 = catheter_3d_system(rand_input2)
    # THEN: outputs are the same
    for key in ['coord1', 'coord2']:
        for (item1, item2) in zip(out1[key], out2[key]):
            assert item1 != item2



