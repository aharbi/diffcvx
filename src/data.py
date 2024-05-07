import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gridstatus

from torch.utils.data import Dataset

from numpy.lib.stride_tricks import sliding_window_view


class CaisoDataset(Dataset):
    def __init__(
        self, data_dir: str, column: str, history_horizon: int, forecast_horizon: int
    ):
        """Generates a PyTorch dataset for a forecasting problem.

        Args:
            data_dir (str): directory where time series dataframe is located.
            column (str): dataframe column for the time series.
            history_horizon (int): length of the prediction history window.
            forecast_horizon (int): length of the prediction future window.
        """
        self.data_dir = data_dir
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon

        self.data = pd.read_csv(data_dir)

        self.time_series = self.data[column].to_numpy()
        self.time_series_shifted = self.time_series[history_horizon:]

        # X, and Y have shape (number of examples, length of horizon window)
        self.X = sliding_window_view(self.time_series, history_horizon)
        self.Y = sliding_window_view(self.time_series_shifted, forecast_horizon)

        self.n = self.Y.shape[0]

        # Match number of examples in X and Y.
        self.X = self.X[: self.n, :]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx, :]

        return x, y


def get_caiso(start_date, end_date, save_dir):
    """Generates hourly electrical load data from the California ISO network.

    Args:
        start_date (pandas.Timestamp): start date of generated data.
        end_date (pandas.Timestamp): end date of generated data.
        save_dir (str): directory to the folder where the generated csv will be saved.

    Returns:
        pandas.Dataframe: dataframe of hourly electrical load.
    """
    caiso = gridstatus.CAISO()

    data = caiso.get_load(start=start_date, end=end_date)

    load = data.set_index("Time").resample("h").mean()
    load.drop([load.columns[0], load.columns[1]], axis=1, inplace=True)

    load.to_csv(save_dir)

    return load


if __name__ == "__main__":

    # Training data
    train_start_date = pd.Timestamp("January 1, 2022").normalize()
    train_end_date = pd.Timestamp("June 30, 2023").normalize()
    train_save_dir = "data/caiso_train.csv"

    _ = get_caiso(train_start_date, train_end_date, train_save_dir)

    # Testing data
    test_start_date = pd.Timestamp("July 1, 2023").normalize()
    test_end_date = pd.Timestamp("December 31, 2023").normalize()
    test_save_dir = "data/caiso_test.csv"

    _ = get_caiso(test_start_date, test_end_date, test_save_dir)
