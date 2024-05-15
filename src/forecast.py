import torch

from torch import nn
from collections import OrderedDict


class Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        num_hidden: int,
    ):
        """Creates a feed forward multilayer perceptron (MLP) for use in forecasting algorithms.

        Args:
            input_dim (int): input dimension (i.e., history_horizon).
            output_dim (int): output dimension (i.e., forecast_horizon).
            num_layers (int): number of layers in the MLP.
            num_hidden (int): number of hidden units per layer.
        """
        super(Forecaster, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_layers = num_layers
        self.num_hidden = num_hidden

        self.model = self.create_model()

    def create_model(self):

        layers = OrderedDict()

        layers["input"] = nn.Linear(self.input_dim, self.num_hidden)
        layers["input_activation"] = nn.GELU()

        for hidden_layer_index in range(self.num_layers):
            layers[f"hidden_{hidden_layer_index}"] = nn.Linear(
                self.num_hidden, self.num_hidden
            )
            layers[f"activation_{hidden_layer_index}"] = nn.GELU()

        layers["output"] = nn.Linear(self.num_hidden, self.output_dim)
        layers["output_activation"] = nn.GELU()

        model = nn.Sequential(layers)

        return model

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    history_horizon = 72
    forecast_horizon = 24

    num_layers = 4
    num_hidden = 256

    # Create a simple MLP forecasting model
    forecaster = Forecaster(
        input_dim=history_horizon,
        output_dim=forecast_horizon,
        num_layers=num_layers,
        num_hidden=num_hidden,
    )
