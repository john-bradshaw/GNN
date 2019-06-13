
import typing

import torch.nn as nn
import torch.nn.functional as F


class MlpParams(typing.NamedTuple):
    input_dim: int
    output_dim: int
    hidden_sizes: typing.List[int]



class MLP(nn.Module):
    #todo:consider adding dropout

    def __init__(self, params: MlpParams):
        super(MLP, self).__init__()
        self.params = params

        layer_sizes = [self.params.input_dim] + self.params.hidden_sizes + [self.params.output_dim]
        layer_dims = zip(layer_sizes[:-1], layer_sizes[1:])

        self.linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for input_dim, output_dim in layer_dims])

    def forward(self, input_tensor):
        hidden = input_tensor
        for i, layer in enumerate(self.linears):
            hidden = layer(hidden)
            if i < self.num_layers - 1:
                hidden = F.relu(hidden)
        return hidden

    @property
    def num_layers(self):
        return len(self.linears)
