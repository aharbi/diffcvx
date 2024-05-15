import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from forecast import Forecaster
from data import ForecastDataset


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
