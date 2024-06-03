import torch
import json

from torch import nn
from torch.utils.data import DataLoader

from cvxpylayers.torch import CvxpyLayer

from tqdm import tqdm

from forecast import Forecaster
from data import ForecastDataset
from model import EconomicDispatchModel


class IndependentTrainer:
    def __init__(self, forecaster: Forecaster, dataset: ForecastDataset):
        """_summary_

        Args:
            forecaster (Forecaster): _description_
            dataset (ForecastDataset): _description_
        """

        self.forecaster = forecaster
        self.dataset = dataset

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        lr: float,
        device: str,
        save_dir: str,
        name: str,
    ):
        """_summary_

        Args:
            num_epochs (int): _description_
            batch_size (int): _description_
            lr (float): _description_
            device (str): _description_
            save_dir (str): _description_
            name (str): _description_
        """

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self.forecaster.parameters(), lr=lr)
        loss = nn.MSELoss()
        loss_log = []

        self.forecaster.to(device)
        self.forecaster.train()

        for epoch in range(num_epochs):

            optim.param_groups[0]["lr"] = lr * (1 - epoch / num_epochs)

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

                loss_log.append(output.item())

                if loss_ema is None:
                    loss_ema = output.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * output.item()

                pbar.set_description("Loss: {:.4f} - Epoch: {}".format(loss_ema, epoch))
                optim.step()

        if not (save_dir is None):
            torch.save(
                self.forecaster.state_dict(),
                save_dir + f"{name}.pth",
            )

        with open(save_dir + f"{name}_loss_log.json", "w") as file:
            json.dump(loss_log, file)


class EndToEndTrainer:
    def __init__(
        self,
        forecaster: Forecaster,
        ed_model: EconomicDispatchModel,
        dataset: ForecastDataset,
    ):
        """_summary_

        Args:
            forecaster (Forecaster): _description_
            ed_model (EconomicDispatchModel): _description_
            dataset (ForecastDataset): _description_
        """

        self.forecaster = forecaster
        self.ed_model = ed_model
        self.dataset = dataset

        self.ed_layer = CvxpyLayer(
            self.ed_model.model,
            parameters=[self.ed_model.d],
            variables=[self.ed_model.g],
        )

    def train(
        self,
        loss: str,
        num_epochs: int,
        batch_size: int,
        lr: float,
        device: str,
        save_dir: str,
        name: str,
    ):
        """_summary_

        Args:
            loss (str): _description_
            num_epochs (int): _description_
            batch_size (int): _description_
            lr (float): _description_
            device (str): _description_
            save_dir (str): _description_
            name (str): _description_
        """

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self.forecaster.parameters(), lr=lr)
        loss_log = []

        self.forecaster.to(device)
        self.forecaster.train()

        for epoch in range(num_epochs):

            optim.param_groups[0]["lr"] = lr * (1 - epoch / num_epochs)

            pbar = tqdm(dataloader)

            loss_ema = None
            for x, y in pbar:
                optim.zero_grad()

                x = x.to(device)
                y = y.to(device)

                y_hat = self.forecaster(x)

                # ECOS seems to be faster for this model.
                g = self.ed_layer(
                    y_hat,
                    solver_args={"solve_method": "ECOS"},
                )

                if loss == "prediction_error":
                    output = prediction_error_loss(g[0], y)

                elif loss == "capex":
                    output = capex_loss(g[0], y, self.ed_model)

                elif loss == "ramping_reserve":
                    ramping_reserve_weight = 6000

                    capex = capex_loss(g[0], y, self.ed_model)
                    ramping_reserve = ramping_reserve_loss(g[0], self.ed_model)

                    output = -1 * ramping_reserve_weight * ramping_reserve + capex

                output.backward()

                loss_ema = output.item()

                loss_log.append(output.item())

                if loss_ema is None:
                    loss_ema = output.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * output.item()

                pbar.set_description(
                    "Loss: {:.4f} - Epoch: {}".format(loss_ema / 8, epoch)
                )
                optim.step()

        if not (save_dir is None):
            torch.save(
                self.forecaster.state_dict(),
                save_dir + f"{name}.pth",
            )

        with open(save_dir + f"{name}_loss_log.json", "w") as file:
            json.dump(loss_log, file)


def prediction_error_loss(g: torch.Tensor, y: torch.Tensor):
    """_summary_

    Args:
        g (torch.Tensor): _description_
        y (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """

    d = g.sum(axis=-2)

    mse = ((d - y) ** 2).mean()

    return mse


def capex_loss(g: torch.Tensor, y: torch.Tensor, ed_model: EconomicDispatchModel):
    """_summary_

    Args:
        g (torch.Tensor): _description_
        y (torch.Tensor): _description_
        ed_model (EconomicDispatchModel): _description_

    Returns:
        _type_: _description_
    """

    d = g.sum(axis=-2)

    num_generators = len(ed_model.generators)
    capex = 0

    for generator_index in range(num_generators):
        capex += ed_model.generators[generator_index].compute_cost(
            g[:, generator_index, :]
        )

    diff = d - y

    diff_under = (d - y) <= 0
    diff_over = (d - y) > 0

    reserve = (
        ed_model.lambda_minus * (diff[diff_under]).abs().sum()
        + ed_model.lambda_plus * (diff[diff_over]).sum()
    )

    capex += reserve

    return capex


def ramping_reserve_loss(g: torch.Tensor, ed_model: EconomicDispatchModel):
    """_summary_

    Args:
        g (torch.Tensor): _description_
        ed_model (EconomicDispatchModel): _description_

    Returns:
        _type_: _description_
    """

    num_generators = len(ed_model.generators)
    ramping_reserve = 0

    for i, generator in enumerate(ed_model.generators):
        for t in range(1, ed_model.horizon):
            ramping_reserve += generator.ramp_rate - torch.abs(
                g[:, i, t] - g[:, i, t - 1]
            )

    average_ramping_reserve = (ramping_reserve / num_generators).sum()

    return average_ramping_reserve
