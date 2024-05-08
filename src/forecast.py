import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

from collections import OrderedDict
from data import ForecastDataset
from tqdm import tqdm


class Forecaster(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_layers: int, num_hidden: int
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
        layers["input_activation"] = nn.ReLU()

        for hidden_layer_index in range(self.num_layers):
            layers[f"hidden_{hidden_layer_index}"] = nn.Linear(
                self.num_hidden, self.num_hidden
            )
            layers[f"activation_{hidden_layer_index}"] = nn.ReLU()

        layers["output"] = nn.Linear(self.num_hidden, self.output_dim)

        model = nn.Sequential(layers)

        return model

    def train(
        self,
        training_data: str,
        column: str,
        num_epochs: int,
        batch_size: int,
        lr: float,
        device: str,
        save_dir: str,
    ):

        training_dataset = ForecastDataset(
            data_dir=training_data,
            column=column,
            history_horizon=self.input_dim,
            forecast_horizon=self.output_dim,
        )

        dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss = nn.MSELoss()

        self.model.to(device)
        self.model.train()

        for epoch in range(num_epochs):

            pbar = tqdm(dataloader)

            loss_ema = None
            for x, y in pbar:
                optim.zero_grad()

                x = x.to(device)
                y = y.to(device)

                y_hat = self.model(x)

                output = loss(y_hat, y)

                output.backward()

                loss_ema = output.item()

                if loss_ema is None:
                    loss_ema = output.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * output.item()

                pbar.set_description("Loss: {:.4f}".format(loss_ema))
                optim.step()

        if not (save_dir is None):
            torch.save(
                self.model.state_dict(),
                save_dir + "mlp_{}.pth".format(epoch),
            )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    history_horizon = 72
    forecast_horizon = 24

    num_layers = 4
    num_hidden = 256

    num_epochs = 50
    batch_size = 128
    lr = 1e-4
    device = "cpu"

    training_dir = "data/caiso_train.csv"
    testing_dir = "data/caiso_test.csv"

    column = "Load"
    save_dir = "models/"

    # Train a simple MLP forecasting model
    forecaster = Forecaster(
        input_dim=history_horizon,
        output_dim=forecast_horizon,
        num_layers=num_layers,
        num_hidden=num_hidden,
    )

    forecaster.train(
        training_data=training_dir,
        column=column,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        save_dir=save_dir,
    )

    # Plot the results on a single testing example
    testing_dataset = ForecastDataset(
        data_dir=testing_dir,
        column=column,
        history_horizon=history_horizon,
        forecast_horizon=forecast_horizon,
    )

    x, y = testing_dataset.__getitem__(500)

    y_hat = forecaster(torch.from_numpy(x)).detach().numpy()

    plt.plot(y_hat, label="Prediction")
    plt.plot(y, label="Ground Truth")
    plt.legend()
    plt.show()
