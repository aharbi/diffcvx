import torch
import numpy as np

from typing import Callable

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EconomicDispatchModel
from forecast import Forecaster
from data import ForecastDataset


def compute_capex_metric(
    forecaster: Forecaster,
    model: EconomicDispatchModel,
    training_dataset: ForecastDataset,
    testing_dataset: ForecastDataset,
    metric: Callable,
    alpha: tuple[float, float],
):
    score = np.zeros(len(testing_dataset))

    dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

    pbar = tqdm(dataloader)

    for i, (x, y) in enumerate(pbar):

        y_hat = forecaster(x)

        y = y.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()

        score[i] = metric(model, y_hat[0, :], y, alpha[0], alpha[1])

    return score.mean(), score.std()


def compute_prediction_metric(
    forecaster: Forecaster,
    testing_dataset: ForecastDataset,
    metric: Callable,
):

    score = torch.zeros(len(testing_dataset))

    dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

    pbar = tqdm(dataloader)

    for i, (x, y) in enumerate(pbar):

        y_hat = forecaster(x)

        score[i] = metric(y_hat, y)

    score = score.detach().cpu().numpy()

    return score.mean(), score.std()


def capex(
    model: EconomicDispatchModel,
    predicted_demand: np.ndarray,
    true_demand: np.ndarray,
    alpha_over: float,
    alpha_under: float,
):

    system_cost = model.solve_ed(predicted_demand)

    diff = predicted_demand - true_demand

    over_cost = (predicted_demand - true_demand)[diff >= 0] * alpha_over
    under_cost = (predicted_demand - true_demand)[diff < 0] * alpha_under

    capex = system_cost + over_cost.sum() + under_cost.sum()

    return capex


def squared_error(y_hat, y):
    return torch.norm(y_hat - y, p=2)


def absolute_error(y_hat, y):
    return torch.norm(y_hat - y, p=1)
