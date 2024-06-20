import torch
import numpy as np

from typing import Callable

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EconomicDispatchModel
from forecast import Forecaster
from data import ForecastDataset


def compute_average_prediction(forecaster: Forecaster, dataset: ForecastDataset):
    """Computes the average prediction of energy demand over a period using a Forecaster on a given dataset.

    Args:
        forecaster (Forecaster): The forecasting model used to predict energy demand.
        dataset (ForecastDataset): The dataset containing energy demand data.

    Returns:
        tuple: A tuple containing mean predictions, standard deviation of predictions, and mean true values.
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
    """Calculates the expected capital expenditure (CAPEX) over a dataset, taking into account overestimation and underestimation costs.

    Args:
        forecaster (Forecaster): The forecasting model used to predict energy demand.
        model (EconomicDispatchModel): The economic dispatch model used for cost calculations.
        dataset (ForecastDataset): The dataset containing energy demand data.
        lambda_plus (float): The penalty cost per unit of energy for overestimation.
        lambda_minus (float): The penalty cost per unit of energy for underestimation.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the CAPEX scores.
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
    """Computes the ramping reserve score, measuring the ability of a power system to handle ramping based on predicted demand.

    Args:
        forecaster (Forecaster): The forecasting model used to predict energy demand.
        model (EconomicDispatchModel): The economic dispatch model used for ramping calculations.
        dataset (ForecastDataset): The dataset containing energy demand data.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the ramping reserve scores.
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
    """Computes a specified prediction metric over a testing dataset using the given forecaster.

    Args:
        forecaster (Forecaster): The forecasting model used for predictions.
        testing_dataset (ForecastDataset): The dataset containing testing data.
        metric (Callable): The metric function used to evaluate prediction accuracy.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the computed scores.
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
    """Calculates the capital expenditure based on the difference between predicted and true energy demand, considering over and under generation costs.

    Args:
        model (EconomicDispatchModel): The economic dispatch model to calculate system costs.
        predicted_demand (np.ndarray): Array of predicted energy demands.
        true_demand (np.ndarray): Array of actual energy demands.
        lambda_plus (float): The penalty cost per unit of energy for overestimation.
        lambda_minus (float): The penalty cost per unit of energy for underestimation.

    Returns:
        float: The total capital expenditure calculated.
    """
    system_cost = model.solve_ed(predicted_demand)

    diff = predicted_demand - true_demand

    over_cost = (predicted_demand - true_demand)[diff >= 0] * lambda_plus
    under_cost = (predicted_demand - true_demand)[diff < 0] * lambda_minus

    capex = system_cost + over_cost.sum() + np.abs(under_cost).sum()

    return capex


def ramping_reserve(model: EconomicDispatchModel, predicted_demand: np.ndarray):
    """Computes the ramping reserve of a system based on predicted energy demand.

    Args:
        model (EconomicDispatchModel): The economic dispatch model used for ramping calculations.
        predicted_demand (np.ndarray): The predicted energy demand.

    Returns:
        float: The calculated average ramping reserve.
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
    """Computes the squared error between predicted and actual values.

    Args:
        y_hat (torch.Tensor): Predicted values.
        y (torch.Tensor): Actual values.

    Returns:
        torch.Tensor: The squared error.
    """
    return torch.norm(y_hat - y, p=2)


def absolute_error(y_hat, y):
    """Computes the absolute error between predicted and actual values.

    Args:
        y_hat (torch.Tensor): Predicted values.
        y (torch.Tensor): Actual values.

    Returns:
        torch.Tensor: The absolute error.
    """
    return torch.norm(y_hat - y, p=1)
