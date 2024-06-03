import torch
import numpy as np

from typing import Callable

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EconomicDispatchModel
from forecast import Forecaster
from data import ForecastDataset


def compute_average_prediction(forecaster, dataset):
    """_summary_

    Args:
        forecaster (_type_): _description_
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_pred = []
    y_true = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (x, y) in enumerate(dataloader):

        if i % 24 == 0:
            y_hat = forecaster(x)

            y_pred.append(y_hat.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

    mean_true = np.array(y_true).reshape(-1, 24).mean(axis=0)

    mean = np.array(y_pred).reshape(-1, 24).mean(axis=0)
    std = np.array(y_pred).reshape(-1, 24).std(axis=0)

    return mean, std, mean_true


def compute_capex(
    forecaster: Forecaster,
    model: EconomicDispatchModel,
    dataset: ForecastDataset,
    lambda_plus: float,
    lambda_minus: float,
):
    """_summary_

    Args:
        forecaster (Forecaster): _description_
        model (EconomicDispatchModel): _description_
        dataset (ForecastDataset): _description_
        lambda_plus (float): _description_
        lambda_minus (float): _description_

    Returns:
        _type_: _description_
    """
    score = np.zeros(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    pbar = tqdm(dataloader)

    for i, (x, y) in enumerate(pbar):

        y_hat = forecaster(x)

        y = y.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()

        score[i] = capex(model, y_hat[0, :], y, lambda_plus, lambda_minus)

    return score.mean(), score.std()


def compute_ramping_reserve(
    forecaster: Forecaster,
    model: EconomicDispatchModel,
    dataset: ForecastDataset,
):
    """_summary_

    Args:
        forecaster (Forecaster): _description_
        model (EconomicDispatchModel): _description_
        dataset (ForecastDataset): _description_

    Returns:
        _type_: _description_
    """
    score = np.zeros(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    pbar = tqdm(dataloader)

    for i, (x, y) in enumerate(pbar):

        y_hat = forecaster(x)

        y_hat = y_hat.detach().cpu().numpy()

        score[i] = ramping_reserve(model, y_hat[0, :])

    return score.mean(), score.std()


def compute_prediction_metric(
    forecaster: Forecaster,
    testing_dataset: ForecastDataset,
    metric: Callable,
):
    """_summary_

    Args:
        forecaster (Forecaster): _description_
        testing_dataset (ForecastDataset): _description_
        metric (Callable): _description_

    Returns:
        _type_: _description_
    """
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
    lambda_plus: float,
    lambda_minus: float,
):
    """_summary_

    Args:
        model (EconomicDispatchModel): _description_
        predicted_demand (np.ndarray): _description_
        true_demand (np.ndarray): _description_
        lambda_plus (float): _description_
        lambda_minus (float): _description_

    Returns:
        _type_: _description_
    """
    system_cost = model.solve_ed(predicted_demand)

    diff = predicted_demand - true_demand

    over_cost = (predicted_demand - true_demand)[diff >= 0] * lambda_plus
    under_cost = (predicted_demand - true_demand)[diff < 0] * lambda_minus

    capex = system_cost + over_cost.sum() + np.abs(under_cost).sum()

    return capex


def ramping_reserve(model: EconomicDispatchModel, predicted_demand: np.ndarray):
    """_summary_

    Args:
        model (EconomicDispatchModel): _description_
        predicted_demand (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    model.solve_ed(predicted_demand)

    num_generators = len(model.generators)
    average_ramping_reserve = 0

    generation = model.g.value

    for i in range(num_generators):
        for t in range(1, model.horizon):
            average_ramping_reserve += model.generators[i].ramp_rate - np.abs(
                generation[i, t] - generation[i, t - 1]
            )

    average_ramping_reserve = average_ramping_reserve / num_generators

    return average_ramping_reserve


def squared_error(y_hat, y):
    """_summary_

    Args:
        y_hat (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.norm(y_hat - y, p=2)


def absolute_error(y_hat, y):
    """_summary_

    Args:
        y_hat (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.norm(y_hat - y, p=1)
