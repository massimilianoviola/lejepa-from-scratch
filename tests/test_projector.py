import torch

from lejepa.projector import Projector


def test_three_layer_projector() -> None:
    projector = Projector(192, (512, 512, 64))
    assert projector(torch.randn(8, 192)).shape == (8, 64)
