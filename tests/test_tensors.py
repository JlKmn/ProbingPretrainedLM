from torch import tensor

def test_tensor_comparison():
    assert tensor(9) == 9
    assert tensor(9) != 0
    assert tensor(9) != 10
    assert tensor(9) != -1