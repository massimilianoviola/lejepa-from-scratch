from collections.abc import Callable, Sequence

import torch


class Projector(torch.nn.Sequential):
    """Simple MLP projection with a variable number of layers."""

    def __init__(
        self,
        in_channels: int = 192,
        dims: Sequence[int] = (64,),
        norm: Callable[..., torch.nn.Module] = torch.nn.BatchNorm1d,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        dropout: float = 0.0,
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in dims[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(norm(hidden_dim))
            layers.append(activation())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, dims[-1]))

        super().__init__(*layers)
