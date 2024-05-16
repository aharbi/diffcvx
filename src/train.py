import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

from cvxpylayers.torch import CvxpyLayer

from tqdm import tqdm

from forecast import Forecaster
from data import ForecastDataset
from model import EconomicDispatchModel


class IndependentTrainer:
    def __init__(self, forecaster: Forecaster, dataset: ForecastDataset):

        self.forecaster = forecaster
        self.dataset = dataset

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        lr: float,
        device: str,
        save_dir: str,
    ):

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self.forecaster.parameters(), lr=lr)
        loss = nn.MSELoss()

        self.forecaster.to(device)
        self.forecaster.train()

        for epoch in range(num_epochs):

            pbar = tqdm(dataloader)

            loss_ema = None
            for x, y in pbar:
                optim.zero_grad()

                x = x.to(device)
                y = y.to(device)

                y_hat = self.forecaster(x)

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
                self.forecaster.state_dict(),
                save_dir + "mlp_{}.pth".format(epoch),
            )


class EndToEndTrainer:
    def __init__(
        self,
        forecaster: Forecaster,
        ed_model: EconomicDispatchModel,
        dataset: ForecastDataset,
    ):

        self.forecaster = forecaster
        self.ed_model = ed_model
        self.dataset = dataset

        self.ed_layer = CvxpyLayer(
            self.ed_model.model,
            parameters=[self.ed_model.d],
            variables=[
                self.ed_model.g,
                self.ed_model.d_over,
                self.ed_model.d_under,
            ],
        )

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        lr: float,
        device: str,
        save_dir: str,
    ):

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self.forecaster.parameters(), lr=lr)
        loss = nn.MSELoss()

        self.forecaster.to(device)
        self.forecaster.train()

        for epoch in range(num_epochs):

            pbar = tqdm(dataloader)

            loss_ema = None
            for x, y in pbar:
                optim.zero_grad()

                x = x.to(device)
                y = y.to(device)

                y_hat = self.forecaster(x)

                x = self.ed_layer(y_hat, solver_args={"eps": 1e-3, "max_iters": 1000, "acceleration_lookback": 0})

                # NOTE: Temporary loss, to be changed.
                d_hat = x[0].sum(axis=-2)

                output = loss(d_hat, y)

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
                    self.forecaster.state_dict(),
                    save_dir + "mlp_{}.pth".format(epoch),
                )
